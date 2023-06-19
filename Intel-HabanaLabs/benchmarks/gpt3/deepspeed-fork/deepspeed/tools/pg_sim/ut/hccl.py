import unittest
from pg_sim.ut.base import TestBaseWrapper


class ProcessGroupSimTestHccl(TestBaseWrapper.ProcessGroupSimTestBase):

    def setUp(self) -> None:
        super(ProcessGroupSimTestHccl, self).setUp()

    def get_backend(self):
        return 'hccl'

    def get_device(self):
        return 'hpu'


if __name__ == '__main__':
    unittest.main()
