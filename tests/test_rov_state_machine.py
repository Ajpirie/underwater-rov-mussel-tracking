"""Deterministic simulation tests; no FilterPy, GUI or ROV hardware required."""
import unittest

from Integration.rov_state_machine import ROVController, ROVState
from Integration.rov_simulator import SimulatedROV


TARGET = (0.1, 0.0, 0.4)


class ROVControllerTests(unittest.TestCase):
    def make_controller(self, **adapter_kwargs):
        adapter = SimulatedROV(**adapter_kwargs)
        return ROVController(adapter, ack_timeout_s=.5, action_timeout_s=5), adapter

    def test_requires_operator_arm(self):
        control, adapter = self.make_controller()
        with self.assertRaises(RuntimeError):
            control.start_approach(TARGET, now_s=0)
        self.assertEqual(adapter.commands, [])
        control.arm()
        control.start_approach(TARGET, now_s=0)
        self.assertEqual(control.state, ROVState.APPROACH)

    def test_approval_waits_for_arrival(self):
        control, adapter = self.make_controller()
        control.arm()
        control.start_approach(TARGET, now_s=0)
        control.tick(.2)
        with self.assertRaises(RuntimeError):
            control.approve_collection(now_s=.2)
        self.assertFalse(any(c[0] == 'collect' for c in adapter.commands))
        control.tick(2.1)
        self.assertTrue(control.arrived)
        self.assertEqual(control.state, ROVState.APPROACH)
        self.assertEqual([x[0] for x in adapter.commands], ['approach'])
        control.approve_collection(now_s=2.1)
        control.tick(2.3)
        self.assertEqual(control.state, ROVState.COLLECT)
        control.tick(3.2)
        self.assertEqual(control.state, ROVState.IDLE)
        self.assertFalse(control.armed)
        self.assertEqual([x[0] for x in adapter.commands], ['approach', 'collect'])

    def test_waits_for_manual_collection_approval_indefinitely(self):
        control, adapter = self.make_controller()
        control.arm()
        control.start_approach(TARGET, now_s=0)
        control.tick(.2)
        control.tick(2.1)
        control.tick(100)
        self.assertEqual(control.state, ROVState.APPROACH)
        self.assertTrue(control.arrived)
        self.assertFalse(any(x[0] == 'collect' for x in adapter.commands))

    def test_stop_during_collection_prevents_completion(self):
        control, adapter = self.make_controller()
        control.arm()
        control.start_approach(TARGET, now_s=0)
        control.tick(.2)
        control.tick(2.1)
        control.approve_collection(now_s=2.1)
        control.stop('Operator stop')
        control.tick(10)
        self.assertEqual(control.state, ROVState.FAULT)
        self.assertEqual(adapter.commands[-1][0], 'stop')

    def test_stops_and_latches_fault_until_reset(self):
        control, adapter = self.make_controller()
        control.arm()
        control.start_approach(TARGET, now_s=0)
        control.stop('Operator stop')
        self.assertEqual(control.state, ROVState.FAULT)
        self.assertEqual(adapter.commands[-1][0], 'stop')
        control.tick(3)
        self.assertFalse(any(x[0] == 'collect' for x in adapter.commands))
        with self.assertRaises(RuntimeError):
            control.arm()
        control.reset()
        self.assertEqual(control.state, ROVState.IDLE)
        self.assertFalse(control.armed)

    def test_rejected_command_faults(self):
        control, adapter = self.make_controller(accept_commands=False)
        control.arm()
        control.start_approach(TARGET, now_s=0)
        control.tick(.2)
        self.assertEqual(control.state, ROVState.FAULT)
        self.assertIn('rejected', control.fault_reason)
        self.assertEqual(adapter.commands[-1][0], 'stop')

    def test_ack_timeout_faults(self):
        control, adapter = self.make_controller(ack_delay_s=2)
        control.arm()
        control.start_approach(TARGET, now_s=0)
        control.tick(.6)
        self.assertEqual(control.state, ROVState.FAULT)
        self.assertIn('acknowledgement', control.fault_reason)

    def test_approach_timeout_faults(self):
        control, adapter = self.make_controller(report_arrival=False)
        control.arm()
        control.start_approach(TARGET, now_s=0)
        control.tick(.2)
        control.tick(5.1)
        self.assertEqual(control.state, ROVState.FAULT)
        self.assertIn('completion', control.fault_reason)

    def test_collection_timeout_faults(self):
        control, adapter = self.make_controller(report_completion=False)
        control.arm()
        control.start_approach(TARGET, now_s=0)
        control.tick(.2)
        control.tick(2.1)
        control.approve_collection(now_s=2.1)
        control.tick(2.3)
        control.tick(7.2)
        self.assertEqual(control.state, ROVState.FAULT)
        self.assertEqual(adapter.commands[-1][0], 'stop')

    def test_invalid_target_cannot_start(self):
        control, adapter = self.make_controller()
        control.arm()
        for target in [(float('nan'), 0, 1), (0, 1), None]:
            with self.assertRaises(ValueError):
                control.start_approach(target, now_s=0)
        self.assertEqual(control.state, ROVState.IDLE)

    def test_simulator_is_separate_from_rov_placeholders(self):
        from Integration import rov_interface
        control, adapter = self.make_controller()
        control.arm()
        control.start_approach(TARGET, now_s=0)
        self.assertNotEqual(adapter.request_approach, rov_interface.CommandROVToTarget)
        self.assertNotEqual(adapter.request_collection, rov_interface.commandROVToCollectMussels)


if __name__ == '__main__':
    unittest.main()
