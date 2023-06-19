import unittest
from pg_sim.ut.base import TestBaseWrapper


class ProcessGroupSimTestGloo(TestBaseWrapper.ProcessGroupSimTestBase):

    def setUp(self) -> None:
        super(ProcessGroupSimTestGloo, self).setUp()

    def get_backend(self):
        return 'gloo'

    def get_device(self):
        return 'cpu'


if __name__ == '__main__':
    unittest.main()
