from copy import deepcopy
from dataclasses import replace
import json
import unittest

from aicf import run
from aicf.backend import execute, select_implementations, transform
from aicf.cuda import lookup
from aicf.operators import Graph, SEMANTICS
from demo import tiny_model


class PipelineTests(unittest.TestCase):
    def test_model_to_graph_to_reference_and_cuda_interface(self):
        report = run(tiny_model(), [[1, 2], [3, 4]])
        compute = lambda graph: [node['type'] for node in graph['nodes'] if node['inputs']]
        self.assertEqual(compute(report['original']), ['matmul', 'add', 'relu', 'matmul', 'add'])
        self.assertEqual(compute(report['transformed']), ['linear_relu', 'matmul', 'add'])
        self.assertEqual(report['execution']['original_output'], [[3], [3]])
        self.assertEqual(report['execution']['transformed_output'], [[3], [3]])
        self.assertTrue(report['execution']['equal'])
        self.assertEqual(report['decisions'][0]['status'], 'ACCEPT')
        self.assertEqual([entry['implementation_id'] for entry in report['cuda']],
                         ['cuda.linear_relu.placeholder', 'cuda.matmul.placeholder', 'cuda.add.placeholder'])
        for entry in report['cuda']:
            self.assertEqual(entry['launch']['status'], 'not-implemented')
            self.assertIsNone(entry['launch']['output'])
            self.assertEqual(entry['launch']['observations'], [])
        self.assertEqual(json.loads(json.dumps(report))['schema_version'], 1)

    def test_semantics_have_scoped_linearity_and_conditional_homogeneity(self):
        linear = [prop for prop in SEMANTICS['matmul'].properties if prop.kind == 'LINEAR']
        self.assertEqual([(prop.argument, prop.condition) for prop in linear],
                         [(0, 'argument 1 fixed'), (1, 'argument 0 fixed')])
        homogeneous = SEMANTICS['relu'].properties[-1]
        self.assertEqual((homogeneous.kind, homogeneous.argument, homogeneous.condition),
                         ('POSITIVE_HOMOGENEOUS', 0, 'scalar >= 0'))
        self.assertIn('ASSOCIATIVE', [prop.kind for prop in SEMANTICS['add'].properties])
        self.assertIn('REDUCTION', [prop.kind for prop in SEMANTICS['reduce_sum'].properties])

    def test_fusion_is_immutable_and_preserves_multiple_reference_inputs(self):
        original = tiny_model().build((2, 2))
        snapshot = deepcopy(original)
        fused, _ = transform(original)
        self.assertEqual(original, snapshot)
        for value in ([[0, 0], [-1, 2]], [[2, -4], [1.5, 2.5]], [[10, 20], [0, -5]]):
            self.assertEqual(execute(original, {'x': value}), execute(fused, {'x': value}))

    def test_shared_intermediate_is_rejected_and_missing_bias_information_is_unknown(self):
        original = tiny_model().build((2, 2))
        output = original.output
        product = next(node for node in original.nodes if node.type == 'matmul')
        original.add('relu', product.id)
        original.output = output
        unchanged, decisions = transform(original)
        self.assertEqual(decisions[0].status, 'REJECT')
        self.assertEqual(unchanged, original)
        dynamic = tiny_model().build((2, 2))
        bias = next(node for node in dynamic.nodes if node.type == 'constant' and node.shape == (3,))
        dynamic.nodes = [replace(node, type='input', attributes={'name': 'b', 'shape': [3]})
                         if node.id == bias.id else node for node in dynamic.nodes]
        self.assertEqual(transform(dynamic)[1][0].status, 'UNKNOWN')

    def test_positive_scale_accept_reject_unknown_and_numeric_results(self):
        for alpha, expected in [(2, 'ACCEPT'), (0, 'ACCEPT'), (-1, 'REJECT')]:
            with self.subTest(alpha=alpha):
                graph = Graph()
                x = graph.add('input', name='x', shape=(3,))
                a = graph.add('constant', value=alpha)
                scale = graph.add('mul', x, a)
                graph.add('relu', scale)
                result, decisions = transform(graph, 'positive_scale')
                self.assertEqual(decisions[0].status, expected)
                self.assertEqual(execute(graph, {'x': [-3, 0, 4]}), execute(result, {'x': [-3, 0, 4]}))
                if alpha >= 0:
                    self.assertEqual(result.output, scale)
                    self.assertEqual(result.nodes[-1].type, 'mul')
                    unchanged, pending = transform(graph, 'positive_scale', domain='ieee')
                    self.assertEqual(pending[0].status, 'UNKNOWN')
                    self.assertEqual(unchanged, graph)
        self.assertNotEqual(max(0, -1 * 1), -1 * max(0, 1))

    def test_undeclared_relation_is_unknown(self):
        graph = Graph()
        x = graph.add('input', name='x', shape=(2,))
        a = graph.add('constant', value=2)
        scale = graph.add('mul', x, a)
        graph.add('reduce_sum', scale)
        self.assertEqual(transform(graph, 'positive_scale')[1][0].status, 'UNKNOWN')
        self.assertEqual(execute(graph, {'x': [1, 2]}), 6)

    def test_graph_and_input_contracts_fail_explicitly(self):
        graph = tiny_model().build((2, 2))
        with self.assertRaises(ValueError):
            execute(graph, {'x': [[1, 2, 3]]})
        broken = deepcopy(graph)
        broken.nodes.reverse()
        with self.assertRaises(ValueError):
            broken.validate()
        with self.assertRaises(ValueError):
            tiny_model().build((2, 3))
        with self.assertRaises(ValueError):
            transform(graph, rule='unknown')

    def test_cuda_lookup_is_explicit_and_does_not_execute_reference_as_gpu(self):
        graph, _ = transform(tiny_model().build((2, 2)))
        selected = select_implementations(graph)
        for node, implementation in selected:
            self.assertEqual(lookup(node.type), implementation)
            self.assertIsNone(implementation.launch((), {})['output'])
        with self.assertRaises(ValueError):
            lookup('not-registered')


if __name__ == '__main__':
    unittest.main()
