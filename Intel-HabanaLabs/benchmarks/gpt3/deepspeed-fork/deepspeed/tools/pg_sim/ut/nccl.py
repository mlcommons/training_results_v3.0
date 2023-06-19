import unittest
from pg_sim.ut.base import TestBaseWrapper


class ProcessGroupSimTestNccl(TestBaseWrapper.ProcessGroupSimTestBase):

    def setUp(self) -> None:
        super(ProcessGroupSimTestNccl, self).setUp()

    def get_backend(self):
        return 'nccl'

    def get_device(self):
        return 'cuda'


if __name__ == '__main__':
    unittest.main()
