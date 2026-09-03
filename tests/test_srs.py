from __future__ import annotations

from datetime import date, timedelta

from studycards.srs import DEFAULT_EASE_FACTOR, MIN_EASE_FACTOR, ReviewState, schedule


class TestIsDue:
    def test_never_reviewed_is_always_due(self):
        assert ReviewState().is_due(date(2026, 1, 1)) is True

    def test_due_today_is_due(self):
        state = ReviewState(due_date=date(2026, 1, 1))
        assert state.is_due(date(2026, 1, 1)) is True

    def test_future_due_date_is_not_due(self):
        state = ReviewState(due_date=date(2026, 1, 5))
        assert state.is_due(date(2026, 1, 1)) is False

    def test_past_due_date_is_due(self):
        state = ReviewState(due_date=date(2026, 1, 1))
        assert state.is_due(date(2026, 1, 5)) is True


class TestScheduleFirstReviews:
    def test_first_good_review_sets_one_day_interval(self):
        today = date(2026, 1, 1)
        new_state = schedule(ReviewState(), "good", today=today)
        assert new_state.repetitions == 1
        assert new_state.interval_days == 1
        assert new_state.due_date == today + timedelta(days=1)
        assert new_state.last_reviewed == today

    def test_second_good_review_sets_six_day_interval(self):
        today = date(2026, 1, 1)
        state = schedule(ReviewState(), "good", today=today)
        state = schedule(state, "good", today=today + timedelta(days=1))
        assert state.repetitions == 2
        assert state.interval_days == 6

    def test_third_good_review_multiplies_by_ease_factor(self):
        today = date(2026, 1, 1)
        state = schedule(ReviewState(), "good", today=today)
        state = schedule(state, "good", today=today + timedelta(days=1))
        ease_before_third = state.ease_factor
        state = schedule(state, "good", today=today + timedelta(days=7))
        assert state.repetitions == 3
        assert state.interval_days == round(6 * ease_before_third)


class TestAgainResetsProgress:
    def test_again_resets_repetitions_and_interval(self):
        today = date(2026, 1, 1)
        state = schedule(ReviewState(), "good", today=today)
        state = schedule(state, "good", today=today + timedelta(days=1))
        lapsed = schedule(state, "again", today=today + timedelta(days=7))
        assert lapsed.repetitions == 0
        assert lapsed.interval_days == 1
        assert lapsed.due_date == today + timedelta(days=8)

    def test_again_does_not_lower_ease_factor_below_minimum(self):
        state = ReviewState(ease_factor=MIN_EASE_FACTOR)
        lapsed = schedule(state, "again")
        assert lapsed.ease_factor >= MIN_EASE_FACTOR


class TestEaseFactorMovement:
    def test_easy_rating_increases_ease_factor(self):
        state = schedule(ReviewState(), "easy")
        assert state.ease_factor > DEFAULT_EASE_FACTOR

    def test_hard_rating_decreases_ease_factor(self):
        state = schedule(ReviewState(), "hard")
        assert state.ease_factor < DEFAULT_EASE_FACTOR

    def test_ease_factor_never_drops_below_floor(self):
        state = ReviewState(ease_factor=MIN_EASE_FACTOR + 0.01)
        for _ in range(20):
            state = schedule(state, "hard")
        assert state.ease_factor >= MIN_EASE_FACTOR

    def test_repeated_easy_ratings_grow_interval_faster_than_repeated_good(self):
        today = date(2026, 1, 1)

        good_state = ReviewState()
        easy_state = ReviewState()
        for i in range(4):
            when = today + timedelta(days=i * 10)
            good_state = schedule(good_state, "good", today=when)
            easy_state = schedule(easy_state, "easy", today=when)

        assert easy_state.interval_days > good_state.interval_days
