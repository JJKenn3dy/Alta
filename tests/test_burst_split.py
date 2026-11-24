import unittest

from postprocessing import split_into_bursts, BURST_MAX_PARTS


class BurstSplitTests(unittest.TestCase):
    def test_short_text_stays_single(self):
        self.assertEqual(split_into_bursts("Иди нахуй."), ["Иди нахуй."])
        self.assertEqual(split_into_bursts("Нет."), ["Нет."])
        self.assertEqual(split_into_bursts("ку"), ["ку"])

    def test_empty(self):
        self.assertEqual(split_into_bursts(""), [])
        self.assertEqual(split_into_bursts("   "), [])

    def test_two_sentences_long_enough_split(self):
        text = "Если серьёзно - вызывай скорую. Если нет - иди нахуй, сам напросился."
        out = split_into_bursts(text)
        self.assertEqual(len(out), 2)
        self.assertTrue(out[0].startswith("Если серьёзно"))
        self.assertTrue(out[1].startswith("Если нет"))

    def test_single_sentence_no_split(self):
        text = "Северная Корея - зачем тебе Windows 12 когда есть Red Star OS как тебе"
        self.assertEqual(split_into_bursts(text), [text])

    def test_capped_at_max_parts(self):
        text = "Первое предложение тут. Второе предложение тут. Третье вот. Четвёртое здесь тоже."
        out = split_into_bursts(text)
        self.assertLessEqual(len(out), BURST_MAX_PARTS)

    def test_micro_fragment_merges(self):
        # "Да." слишком короткий — склеится с соседом, а не станет отдельным бёрстом
        text = "Да. Это правда всё так и есть на самом деле как ни крути."
        out = split_into_bursts(text)
        self.assertTrue(all(len(c) >= 3 for c in out))


if __name__ == "__main__":
    unittest.main()
