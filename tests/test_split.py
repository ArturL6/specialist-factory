from specialist_factory.schemas import Sample
from specialist_factory.training.engine import split_samples


def test_split_samples_is_deterministic_and_covers_every_id_once():
    samples = [Sample(id=f"s{i}") for i in range(10)]
    train_a, validation_a = split_samples(samples, seed=7)
    train_b, validation_b = split_samples(samples, seed=7)
    assert [s.id for s in train_a] == [s.id for s in train_b]
    assert [s.id for s in validation_a] == [s.id for s in validation_b]
    combined_ids = [s.id for s in train_a] + [s.id for s in validation_a]
    assert sorted(combined_ids) == sorted(s.id for s in samples)
    assert len(set(combined_ids)) == len(samples)
