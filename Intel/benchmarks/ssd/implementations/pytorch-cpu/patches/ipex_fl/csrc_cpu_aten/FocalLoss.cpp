// #include <torch/extension.h>
#include <torch/all.h>
#include <torch/csrc/autograd/function.h>
#include <ATen/ATen.h>
#include "FocalLoss.h"
#include <ATen/core/Reduction.h>
#include <ATen/cpu/vml.h>
#include <ATen/cpu/vec/vec.h>
#include <omp.h>

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>
#include <sleef.h>
#include <immintrin.h>

#define MAX_THREAD_ONE_FOR_DEBUG 0
#define USE_INLINE 1
#define FWD_ORIG 0
#define BWD_ORIG 0

static inline void cvtbf16_fp32(const __m256i& a, __m512& o) {
  o = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(a), 16));
//  o = _mm512_cvtpbh_ps(a); // bf16 instructions are supported from gcc12
}


static inline void cvtbf16_fp32(const __m512i& a, __m512& o1, __m512& o2) {
  __m256i lo = _mm512_extracti32x8_epi32(a, 0);
  __m256i hi = _mm512_extracti32x8_epi32(a, 1);
  cvtbf16_fp32(lo, o1);
  cvtbf16_fp32(hi, o2);
}

static inline __m512i cvtfp32_bf16(const __m512& a, const __m512& b) {
  __m512i lo = _mm512_castps_si512(a);
  __m512i hi = _mm512_castps_si512(b);
  __m512i nan = _mm512_set1_epi32(0xffff);
  auto mask_lo = _mm512_cmp_ps_mask(a, a, _CMP_ORD_Q);
  auto mask_hi = _mm512_cmp_ps_mask(b, b, _CMP_ORD_Q);
  __m512i ones = _mm512_set1_epi32(0x1);
  __m512i vec_bias = _mm512_set1_epi32(0x7fff);
  // uint32_t lsb = (input >> 16) & 1;
  auto t_lo = _mm512_and_si512(_mm512_srli_epi32(lo, 16), ones);
  auto t_hi = _mm512_and_si512(_mm512_srli_epi32(hi, 16), ones);
  // uint32_t rounding_bias = 0x7fff + lsb;
  t_lo = _mm512_add_epi32(t_lo, vec_bias);
  t_hi = _mm512_add_epi32(t_hi, vec_bias);
  // input += rounding_bias;
  t_lo = _mm512_add_epi32(t_lo, lo);
  t_hi = _mm512_add_epi32(t_hi, hi);
  // input = input >> 16;
  t_lo = _mm512_srli_epi32(t_lo, 16);
  t_hi = _mm512_srli_epi32(t_hi, 16);
  // Check NaN before converting back to bf16
  t_lo = _mm512_mask_blend_epi32(mask_lo, nan, t_lo);
  t_hi = _mm512_mask_blend_epi32(mask_hi, nan, t_hi);

  t_lo = _mm512_packus_epi32(t_lo, t_hi); // t_hi[4-7] t_lo[4-7] t_hi[0-4] t_lo[0-4]
  __m512i idx = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
  return _mm512_permutexvar_epi64(idx, t_lo);
}


static inline void inner_focal_loss_forward_loop(__m512& in1_fVec, __m512& in2_fVec,
        __m512& tg1_fVec, __m512& tg2_fVec, __m512& temp1_3, __m512& temp2_3)
{
    auto exp_in1_fVec = Sleef_expf16_u10(in1_fVec);
    auto exp_in2_fVec = Sleef_expf16_u10(in2_fVec);

    __m512 temp1_1, temp1_2, temp1_4, temp1_5;
    __m512 temp2_1, temp2_2, temp2_4, temp2_5;

    // -0.25 * input * target
    __m512 cst = _mm512_set1_ps(-0.25f);
    temp1_1 = _mm512_mul_ps(cst, in1_fVec);
    temp1_1 = _mm512_mul_ps(temp1_1, tg1_fVec);
    temp2_1 = _mm512_mul_ps(cst, in2_fVec);
    temp2_1 = _mm512_mul_ps(temp2_1, tg2_fVec);

    // log( exp(input) + 1)
    cst = _mm512_set1_ps(1.0f);
    temp1_2 = _mm512_add_ps(exp_in1_fVec, cst);
    temp2_2 = _mm512_add_ps(exp_in2_fVec, cst);
    temp1_3 = Sleef_logf16_u10(temp1_2);
    temp2_3 = Sleef_logf16_u10(temp2_2);

    // pow( exp(input), 2)
    cst = _mm512_set1_ps(2.0f);
    temp1_4 = Sleef_powf16_u10(exp_in1_fVec, cst);
    temp2_4 = Sleef_powf16_u10(exp_in2_fVec, cst);

    // (0.75-0.75*tar)
    cst = _mm512_set1_ps(0.75f);
    temp1_5 = _mm512_mul_ps(cst, tg1_fVec); 
    temp1_5 = _mm512_sub_ps(cst, temp1_5);
    temp1_4 = _mm512_mul_ps(temp1_4, temp1_5);
    temp2_5 = _mm512_mul_ps(cst, tg2_fVec); 
    temp2_5 = _mm512_sub_ps(cst, temp2_5);
    temp2_4 = _mm512_mul_ps(temp2_4, temp2_5);


    // 0.25*tar
    cst = _mm512_set1_ps(0.25f);
    temp1_5 = _mm512_mul_ps(cst, tg1_fVec);
    temp2_5 = _mm512_mul_ps(cst, tg2_fVec);

    // 
    temp1_4 = _mm512_add_ps(temp1_4, temp1_5);
    temp1_3 = _mm512_mul_ps(temp1_3, temp1_4);
    temp2_4 = _mm512_add_ps(temp2_4, temp2_5);
    temp2_3 = _mm512_mul_ps(temp2_3, temp2_4);

    // add all together
    temp1_1 = _mm512_add_ps(temp1_1, temp1_3);
    temp2_1 = _mm512_add_ps(temp2_1, temp2_3);

    // divide term
    cst = _mm512_set1_ps(2.0f);
    temp1_2 = Sleef_powf16_u10(temp1_2, cst);
    temp2_2 = Sleef_powf16_u10(temp2_2, cst);

    // division
    temp1_3 = _mm512_div_ps(temp1_1, temp1_2);
    temp2_3 = _mm512_div_ps(temp2_1, temp2_2);

}

static inline void inner_focal_loss_backward_loop(__m512& in1_fVec, __m512& in2_fVec,
        __m512& tg1_fVec, __m512& tg2_fVec, 
        float& grad_out,
        __m512& temp1_1, __m512& temp2_1)
{
    auto exp_in1_fVec = Sleef_expf16_u10(in1_fVec);
    auto exp_in2_fVec = Sleef_expf16_u10(in2_fVec);

    __m512 temp1_2, temp1_3, temp1_4, temp1_5;
    __m512 temp2_2, temp2_3, temp2_4, temp2_5;

    __m512 target_0_5_1, target_0_5_2;

    // [1]: 0.5*target * input * exp(input)
    __m512 cst = _mm512_set1_ps(0.5f);
    temp1_1 = _mm512_mul_ps(cst, tg1_fVec);
    target_0_5_1 = temp1_1;
    temp1_1 = _mm512_mul_ps(temp1_1, in1_fVec);
    temp1_1 = _mm512_mul_ps(temp1_1, exp_in1_fVec);
    temp2_1 = _mm512_mul_ps(cst, tg2_fVec);
    target_0_5_2 = temp2_1;
    temp2_1 = _mm512_mul_ps(temp2_1, in2_fVec);
    temp2_1 = _mm512_mul_ps(temp2_1, exp_in2_fVec);
    
    // [2]: pow(exp(input), 3)
    cst = _mm512_set1_ps(3.0f);
    temp1_2 = Sleef_powf16_u10(exp_in1_fVec, cst);
    temp2_2 = Sleef_powf16_u10(exp_in2_fVec, cst);

    // [3]: (0.75-0.75*tar)
    cst = _mm512_set1_ps(0.75f);
    __m512 ones_fVec = _mm512_set1_ps(1.0f);
    temp1_3 = _mm512_sub_ps(ones_fVec, tg1_fVec);
    temp1_3 = _mm512_mul_ps(cst, temp1_3); 
    temp2_3 = _mm512_sub_ps(ones_fVec, tg2_fVec);
    temp2_3 = _mm512_mul_ps(cst, temp2_3); 

    // [4]: [2]*[3]
    temp1_2 = _mm512_mul_ps(temp1_2, temp1_3);
    temp2_2 = _mm512_mul_ps(temp2_2, temp2_3);
    
    // [5]: [1]+[4]
    temp1_1 = _mm512_add_ps(temp1_1, temp1_2);
    temp2_1 = _mm512_add_ps(temp2_1, temp2_2);

    // [6]: log( exp(input) + 1)
    temp1_5 = _mm512_add_ps(exp_in1_fVec, ones_fVec);
    temp2_5 = _mm512_add_ps(exp_in2_fVec, ones_fVec);
    temp1_2 = Sleef_logf16_u10(temp1_5);
    temp2_2 = Sleef_logf16_u10(temp2_5);

    // [7]: [6] * exp(input)
    temp1_2 = _mm512_mul_ps(temp1_2, exp_in1_fVec);
    temp2_2 = _mm512_mul_ps(temp2_2, exp_in2_fVec);

    // [8]: [3]*2*exp(input) - 0.5*target
    cst = _mm512_set1_ps(2.0f);
    temp1_3 = _mm512_mul_ps(temp1_3, cst);
    temp2_3 = _mm512_mul_ps(temp2_3, cst);
    temp1_3 = _mm512_mul_ps(temp1_3, exp_in1_fVec);
    temp2_3 = _mm512_mul_ps(temp2_3, exp_in2_fVec);
    temp1_3 = _mm512_sub_ps(temp1_3, target_0_5_1);
    temp2_3 = _mm512_sub_ps(temp2_3, target_0_5_2);

    // [9]: [6]*[8]
    temp1_2 = _mm512_mul_ps(temp1_2, temp1_3);
    temp2_2 = _mm512_mul_ps(temp2_2, temp2_3);

    // [10]: [9] -0.25*target
    cst = _mm512_set1_ps(0.25f);
    temp1_4 = _mm512_mul_ps(cst, tg1_fVec);
    temp2_4 = _mm512_mul_ps(cst, tg2_fVec);
    temp1_2 = _mm512_sub_ps(temp1_2, temp1_4);
    temp2_2 = _mm512_sub_ps(temp2_2, temp2_4);

    // [11]: [1]+[10]
    temp1_1 = _mm512_add_ps(temp1_1, temp1_2);
    temp2_1 = _mm512_add_ps(temp2_1, temp2_2);

    // [12]: pow(exp(input)+1, 3)
    cst = _mm512_set1_ps(3.0f);
    temp1_2 = Sleef_powf16_u10(temp1_5, cst);
    temp2_2 = Sleef_powf16_u10(temp2_5, cst);

    // [13]: [11]/[12]
    temp1_1 = _mm512_div_ps(temp1_1, temp1_2);
    temp2_1 = _mm512_div_ps(temp2_1, temp2_2);

    // [14]: [13] * grad_out
    cst = _mm512_set1_ps(grad_out);
    temp1_1 = _mm512_mul_ps(temp1_1, cst);
    temp2_1 = _mm512_mul_ps(temp2_1, cst);
}

static inline void focal_loss_forward_loop(c10::BFloat16* input_data_ptr, 
        c10::BFloat16* target_data_ptr, c10::BFloat16* loss_data_ptr, int64_t len, int max_threads) {
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        const int vec_size = 32;
        int64_t start = len/max_threads * tid;
        int64_t end = len/max_threads * (tid+1);
        if (tid == max_threads-1)
            end = len;
        int64_t len2 = end - start;
        int64_t end2 = end - (len2%vec_size);
        int64_t count = end - end2;
        auto stride = vec_size;

        int i = 0;
        for (i = start ; i < end2 ; i+=stride) {
            __m512 in1_fVec, in2_fVec;
            __m512 tg1_fVec, tg2_fVec;

            auto in_bVec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(input_data_ptr+i));
            cvtbf16_fp32((__m512i)in_bVec, in1_fVec, in2_fVec);

            auto tg_bVec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(target_data_ptr+i));
            cvtbf16_fp32((__m512i)tg_bVec, tg1_fVec, tg2_fVec);

            __m512 res1, res2;
            inner_focal_loss_forward_loop(in1_fVec, in2_fVec, tg1_fVec, tg2_fVec, res1, res2);

            // fp32 -> bf16
            __m512i res = (__m512i)cvtfp32_bf16(res1, res2);
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(loss_data_ptr+i), res);
        }
        if (count > 0) {
            __at_align__ int16_t tmp_values[32] = {0,};
            __m512 in1_fVec, in2_fVec;
            __m512 tg1_fVec, tg2_fVec;

            in1_fVec = _mm512_setzero_ps();
            in2_fVec = _mm512_setzero_ps();
            tg1_fVec = _mm512_setzero_ps();
            tg2_fVec = _mm512_setzero_ps();

            std::memcpy(tmp_values, input_data_ptr+i, count * sizeof(int16_t));
            auto in_bVec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(tmp_values));
            cvtbf16_fp32((__m512i)in_bVec, in1_fVec, in2_fVec);

            std::memcpy(tmp_values, target_data_ptr+i, count * sizeof(int16_t));
            auto tg_bVec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(tmp_values));
            cvtbf16_fp32((__m512i)tg_bVec, tg1_fVec, tg2_fVec);

            __m512 res1, res2;
            inner_focal_loss_forward_loop(in1_fVec, in2_fVec, tg1_fVec, tg2_fVec, res1, res2);

            // fp32 -> bf16
            __m512i res = (__m512i)cvtfp32_bf16(res1, res2);
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(tmp_values), res);
            std::memcpy(loss_data_ptr+i, tmp_values, count * sizeof(int16_t));
        }
    } // OpenMP
}

static inline void focal_loss_backward_loop(c10::BFloat16* input_data_ptr, 
        c10::BFloat16* target_data_ptr, 
        float grad_out, 
        c10::BFloat16* grad_input_data_ptr, int64_t len, int max_threads) {
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        const int vec_size = 32;
        int64_t start = len/max_threads * tid;
        int64_t end = len/max_threads * (tid+1);
        if (tid == max_threads-1)
            end = len;
        int64_t len2 = end - start;
        int64_t end2 = end - (len2%vec_size);
        int64_t count = end - end2;
        auto stride = vec_size;

        int i = 0;
        for (i = start ; i < end2 ; i+=stride) {
            __m512 in1_fVec, in2_fVec;
            __m512 tg1_fVec, tg2_fVec;

            auto in_bVec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(input_data_ptr+i));
            cvtbf16_fp32((__m512i)in_bVec, in1_fVec, in2_fVec);

            auto tg_bVec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(target_data_ptr+i));
            cvtbf16_fp32((__m512i)tg_bVec, tg1_fVec, tg2_fVec);

            __m512 res1, res2;
            inner_focal_loss_backward_loop(in1_fVec, in2_fVec, tg1_fVec, tg2_fVec, grad_out, res1, res2);

            // fp32 -> bf16
            __m512i res = (__m512i)cvtfp32_bf16(res1, res2);
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(grad_input_data_ptr+i), res);
        }
        if (count > 0) {
            __at_align__ int16_t tmp_values[vec_size]={0,};
            __m512 in1_fVec, in2_fVec;
            __m512 tg1_fVec, tg2_fVec;

            in1_fVec = _mm512_setzero_ps();
            in2_fVec = _mm512_setzero_ps();
            tg1_fVec = _mm512_setzero_ps();
            tg2_fVec = _mm512_setzero_ps();

            std::memcpy(tmp_values, input_data_ptr+i, count * sizeof(int16_t));
            auto in_bVec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(tmp_values));
            cvtbf16_fp32((__m512i)in_bVec, in1_fVec, in2_fVec);

            std::memcpy(tmp_values, target_data_ptr+i, count * sizeof(int16_t));
            auto tg_bVec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(tmp_values));
            cvtbf16_fp32((__m512i)tg_bVec, tg1_fVec, tg2_fVec);

            __m512 res1, res2;
            inner_focal_loss_backward_loop(in1_fVec, in2_fVec, tg1_fVec, tg2_fVec, grad_out, res1, res2);

            // fp32 -> bf16
            __m512i res = (__m512i)cvtfp32_bf16(res1, res2);
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(tmp_values), res);
            std::memcpy(grad_input_data_ptr+i, tmp_values, count * sizeof(int16_t));
        }
    } // OpenMP
}


namespace {
  static inline at::Tensor apply_loss_reduction(const at::Tensor& unreduced, int64_t reduction) {
    if (reduction == at::Reduction::Mean) {
      return unreduced.mean();
    } else if (reduction == at::Reduction::Sum) {
      return unreduced.sum();
    }
    return unreduced;
  }
}// namespace


namespace torch_ipex {
    namespace cpu {
        at::Tensor _focal_loss_forward(const at::Tensor& input, const at::Tensor& target, const int64_t reduction) {
            static int count = 0;

            c10::BFloat16* input_data_ptr = (c10::BFloat16*) input.flatten().data_ptr();
            c10::BFloat16* target_data_ptr = (c10::BFloat16*) target.flatten().data_ptr();
            auto sizes = input.sizes();
            int64_t len = sizes[0]*sizes[1];

            int max_threads = omp_get_max_threads();
            at::Tensor loss = at::empty({sizes[0],sizes[1]}, input.dtype()).set_requires_grad(false);

            c10::BFloat16* loss_data_ptr = (c10::BFloat16*) loss.data_ptr();
            focal_loss_forward_loop(input_data_ptr, target_data_ptr, loss_data_ptr, len, max_threads);

            return apply_loss_reduction(loss, reduction);
        }


        at::Tensor _focal_loss_backward(const at::Tensor& grad, const at::Tensor& input, const at::Tensor& target, const int64_t reduction) {
            c10::BFloat16* input_data_ptr = (c10::BFloat16*) input.data_ptr();
            c10::BFloat16* target_data_ptr = (c10::BFloat16*) target.data_ptr();
            auto sizes = input.sizes();
            int64_t len = sizes[0]*sizes[1];

            at::Tensor grad_input = at::empty({sizes[0],sizes[1]}, input.dtype()).set_requires_grad(false);

            int max_threads = omp_get_max_threads();
            float grad_out = static_cast<float>(((c10::BFloat16*)grad.data_ptr())[0]);
            c10::BFloat16* grad_input_data_ptr = (c10::BFloat16*) grad_input.data_ptr();

            focal_loss_backward_loop(input_data_ptr, target_data_ptr, grad_out, grad_input_data_ptr, len, max_threads);

            if (reduction == at::Reduction::Mean) {
                return grad_input / input.numel();
            }

            return grad_input;
        }
    }// cpu
}// namespace torch_ipex


namespace torch_ipex {

    at::Tensor focal_loss_forward(const at::Tensor& input, const at::Tensor& target, const int64_t reduction) {
        //RECORD_FUNCTION("torch_ipex::focal_loss_forward", c10::ArrayRef<c10::IValue>({}));
        return cpu::_focal_loss_forward(input, target, reduction);
    }

    at::Tensor focal_loss_backward(const at::Tensor& grad, const at::Tensor& input, const at::Tensor& target, const int64_t reduction) {
        //RECORD_FUNCTION("torch_ipex::focal_loss_backward", c10::ArrayRef<c10::IValue>({}));
        return cpu::_focal_loss_backward(grad, input, target, reduction);
    }

} // namespace torch_ipex

namespace {
TORCH_LIBRARY_FRAGMENT(torch_ipex, m) {
    m.def("focal_loss_forward", torch_ipex::focal_loss_forward);
    m.def("focal_loss_backward", torch_ipex::focal_loss_backward);
}
}


