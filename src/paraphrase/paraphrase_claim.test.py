import unittest
from transformers import T5Tokenizer
from pandas import DataFrame
from paraphrase_claim import ParaphraseDataset, DataFrameConfig

class TestParaphraseDataset(unittest.TestCase):
    def setUp(self):
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
        self.df = DataFrame({'org_claim': ['The earth is flat', 'Vaccines cause autism'], 
                             'gen_claim': ['The earth is a sphere', 'Vaccines are safe']})
        self.max_len = 512
        self.config = DataFrameConfig()
        self.dataset = ParaphraseDataset(tokenizer=self.tokenizer, dataframe=self.df, max_len=self.max_len, config=self.config)

    def test_len(self):
        self.assertEqual(len(self.dataset), 2)

    def test_getitem(self):
        item = self.dataset[0]
        self.assertIn('source_ids', item)
        self.assertIn('source_mask', item)
        self.assertIn('target_ids', item)
        self.assertIn('target_mask', item)

    def test_build(self):
        self.assertEqual(len(self.dataset.inputs), 2)
        self.assertEqual(len(self.dataset.targets), 2)
        self.assertEqual(self.dataset.inputs[0]['input_ids'].shape, (1, self.max_len))
        self.assertEqual(self.dataset.targets[0]['input_ids'].shape, (1, self.max_len))