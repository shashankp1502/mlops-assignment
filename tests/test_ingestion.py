"""unittest coverage for heart_risk.ingestion -- parsing logic only, no
network calls (fetch_raw_text is a thin requests.get wrapper not worth
mocking at length)."""

import unittest

from heart_risk.ingestion import RAW_COLUMNS, ClevelandHeartDiseaseIngestor

SAMPLE_RAW_TEXT = (
    "63.0,1.0,1.0,145.0,233.0,1.0,2.0,150.0,0.0,2.3,3.0,0.0,6.0,0\n"
    "67.0,1.0,4.0,160.0,286.0,0.0,2.0,108.0,1.0,1.5,2.0,3.0,3.0,2\n"
    "41.0,0.0,2.0,130.0,204.0,0.0,2.0,172.0,0.0,1.4,1.0,?,3.0,0\n"
)


class ParseClevelandTests(unittest.TestCase):
    def setUp(self):
        self.ingestor = ClevelandHeartDiseaseIngestor()

    def test_column_names(self):
        frame = self.ingestor.parse(SAMPLE_RAW_TEXT)
        expected = [c for c in RAW_COLUMNS if c != "num"] + ["target"]
        self.assertListEqual(list(frame.columns), expected)

    def test_row_count(self):
        frame = self.ingestor.parse(SAMPLE_RAW_TEXT)
        self.assertEqual(len(frame), 3)

    def test_question_mark_becomes_nan(self):
        frame = self.ingestor.parse(SAMPLE_RAW_TEXT)
        self.assertTrue(frame.loc[2, "ca"] != frame.loc[2, "ca"])  # NaN check

    def test_target_binarization(self):
        frame = self.ingestor.parse(SAMPLE_RAW_TEXT)
        self.assertListEqual(frame["target"].tolist(), [0, 1, 0])

    def test_num_column_dropped(self):
        frame = self.ingestor.parse(SAMPLE_RAW_TEXT)
        self.assertNotIn("num", frame.columns)


if __name__ == "__main__":
    unittest.main()
