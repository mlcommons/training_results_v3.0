#include <ATen/Tensor.h>

namespace torch_ipex {
namespace cpu{

at::Tensor focal_loss_forward(const at::Tensor& input, const at::Tensor& target, const int64_t reduction);

at::Tensor focal_loss_backward(const at::Tensor& grad, const at::Tensor& input, const at::Tensor& target, const int64_t reduction);

} // namespace cpu
} // namespace torch_ipex

