"""Capacity lower bounds and dtype accounting without loading weight payloads."""
import unittest

from exp_code.positional_intervention.check_loading import (
    InsufficientMemory, require_capacity, summarize_tensors, tensor_record,
)


class LoadingCapacityTests(unittest.TestCase):
    def test_failed_moe_tensor_matches_968_mib_request(self):
        row = tensor_record(['layer_0','mlp','gating_einsum','w'],
                            {'shape':[128,2,704,2816], 'dtype':'bfloat16'})
        self.assertEqual(row['bytes'], 968*2**20)
        self.assertTrue(row['retained_text_only'])

    def test_peak_includes_discarded_multimodal_and_preserves_dtype_sizes(self):
        rows = [tensor_record(['embedder','input_embedding'], {'shape':[4,8], 'dtype':'bfloat16'}),
                tensor_record(['embedder','mm_input_projection','w'], {'shape':[4,8], 'dtype':'<f4'}),
                tensor_record(['vision_encoder','w'], {'shape':[3,8], 'dtype':'bfloat16'})]
        result = summarize_tensors(rows)
        self.assertEqual(result['retained_text_bytes'], 64)
        self.assertEqual(result['total_restore_bytes'], 64+128+48)
        self.assertEqual(result['temporary_multimodal_bytes'], 128+48)

    def test_a6000_rejected_before_restore_even_if_allocator_raised(self):
        measured = {'total_restore_bytes': 51647857308}
        with self.assertRaises(InsufficientMemory):
            require_capacity(measured, 49140*2**20, 48354033664)
        with self.assertRaises(InsufficientMemory):
            require_capacity(measured, 49140*2**20, 49140*2**20)
        require_capacity(measured, 80*2**30, int(.95*80*2**30))

    def test_unknown_allocator_limit_is_not_assumed_unbounded(self):
        with self.assertRaises(InsufficientMemory):
            require_capacity({'total_restore_bytes': 100}, 1000, None)


if __name__ == '__main__':
    unittest.main()
