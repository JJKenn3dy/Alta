import unittest
from datetime import datetime, timedelta, timezone

try:
    from main import (
        _finalize_daily_channel_promo_text,
        _daily_channel_promo_target_dt,
        _daily_channel_promo_window_hours,
        _should_send_daily_channel_promo_now,
    )
except Exception as exc:  # pragma: no cover - import guard for optional deps
    _finalize_daily_channel_promo_text = None
    _daily_channel_promo_target_dt = None
    _daily_channel_promo_window_hours = None
    _should_send_daily_channel_promo_now = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(
    _daily_channel_promo_target_dt is None,
    f"failed to import main daily promo helpers: {IMPORT_ERROR}",
)
class MainDailyPromoTests(unittest.TestCase):
    def test_window_hours_are_normalized(self):
        self.assertEqual(_daily_channel_promo_window_hours(12, 19), (12, 19))
        self.assertEqual(_daily_channel_promo_window_hours(22, 10), (22, 23))
        self.assertEqual(_daily_channel_promo_window_hours(-3, 99), (0, 24))

    def test_target_time_is_stable_and_inside_window(self):
        now = datetime(2026, 3, 3, 13, 0, tzinfo=timezone.utc)
        target_a = _daily_channel_promo_target_dt(123, now, 12, 19)
        target_b = _daily_channel_promo_target_dt(123, now, 12, 19)
        window_start = datetime(2026, 3, 3, 12, 0, tzinfo=timezone.utc)
        window_end = datetime(2026, 3, 3, 19, 0, tzinfo=timezone.utc)

        self.assertEqual(target_a, target_b)
        self.assertGreaterEqual(target_a, window_start)
        self.assertLess(target_a, window_end)

    def test_should_send_only_after_target_and_once_per_day(self):
        now = datetime(2026, 3, 3, 13, 0, tzinfo=timezone.utc)
        target = _daily_channel_promo_target_dt(123, now, 12, 19)
        before_target = max(
            datetime(2026, 3, 3, 12, 0, tzinfo=timezone.utc),
            target - timedelta(minutes=1),
        )
        after_target = target + timedelta(minutes=1)

        self.assertFalse(_should_send_daily_channel_promo_now(before_target, 0.0, 123, 12, 19))
        self.assertTrue(_should_send_daily_channel_promo_now(after_target, 0.0, 123, 12, 19))
        self.assertFalse(
            _should_send_daily_channel_promo_now(
                after_target + timedelta(hours=1),
                after_target.timestamp(),
                123,
                12,
                19,
            )
        )

    def test_finalize_promo_text_keeps_required_link(self):
        value = _finalize_daily_channel_promo_text("Залетай в канал, там апдейты.")
        self.assertIn("Ссылка: https://t.me/AltaTheBot", value)

    def test_finalize_promo_text_strips_service_target_marker(self):
        value = _finalize_daily_channel_promo_text(
            "[to current speaker id=1 @user] У меня канал с апдейтами.\n\nСсылка: https://t.me/AltaTheBot"
        )
        self.assertEqual(
            value,
            "У меня канал с апдейтами.\n\nСсылка: https://t.me/AltaTheBot",
        )


if __name__ == "__main__":
    unittest.main()
