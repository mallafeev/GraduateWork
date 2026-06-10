import pytest
from ml.sequence_processing import SequenceSanitizer, PairwiseDistanceCalculator
from phylo.distance import p_distance
from ml.clade_statistics import SubtreeBalanceCalculator
from Bio.Phylo.BaseTree import Clade

class TestSequenceSanitizer:
    @pytest.fixture
    def sanitizer(self):
        return SequenceSanitizer()

    @pytest.mark.parametrize("input_seq, expected", [
        ("acgt", "ACGT"),
        ("A C G T", "ACGT"),
        ("ACGT123!@#", "ACGT"),
        ("", ""),
        ("acgtNn?- ", "ACGT?-"),
    ])
    def test_sanitize_various_inputs(self, sanitizer, input_seq, expected):
        assert sanitizer.sanitize(input_seq) == expected


class TestPDistance:
    @pytest.mark.parametrize("seq1, seq2, expected", [
        ("ACGT", "ACGT", 0.0),
        ("AAAA", "CCCC", 1.0),
        ("A-GT", "C-GT", 1/3),
    ])
    def test_p_distance_calculation(self, seq1, seq2, expected):
        assert p_distance(seq1, seq2) == pytest.approx(expected, abs=1e-6)


class TestSubtreeBalance:
    @pytest.fixture
    def calculator(self):
        return SubtreeBalanceCalculator()

    def _make_clade(self, n_left, n_right):
        root = Clade()
        left = Clade()
        right = Clade()
        for i in range(n_left):
            left.clades.append(Clade(name=f"L{i}"))
        for i in range(n_right):
            right.clades.append(Clade(name=f"R{i}"))
        root.clades = [left, right]
        return root

    def test_symmetric_balance(self, calculator):
        clade = self._make_clade(4, 4)
        assert calculator.calculate(clade) == pytest.approx(0.0)

    def test_asymmetric_balance(self, calculator):
        clade = self._make_clade(3, 1)
        # |3-1| / (3+1) = 0.5
        assert calculator.calculate(clade) == pytest.approx(0.5)

    def test_single_child_balance(self, calculator):
        clade = Clade()
        clade.clades.append(Clade(name="OnlyOne"))
        assert calculator.calculate(clade) == pytest.approx(0.0)