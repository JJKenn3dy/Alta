import unittest

from user_profile import (
    apply_profile_updates,
    delete_profile_field,
    format_profile_view,
    parse_profile_command,
)


class UserProfileCommandTests(unittest.TestCase):
    def test_parse_patch_command(self):
        command = parse_profile_command(
            "/profile patch name=Иван; gender=мужской; interests=музыка, код; city=Москва"
        )

        self.assertIsNotNone(command)
        self.assertEqual(command["action"], "patch")
        self.assertEqual(
            [item["field"] for item in command["updates"]],
            ["approved_display_name", "gender", "interests", "city"],
        )

    def test_apply_updates_sets_name_and_extra_fields(self):
        updated = apply_profile_updates(
            {},
            [
                {
                    "field": "approved_display_name",
                    "kind": "column",
                    "label": "Имя",
                    "value": "Иван",
                },
                {
                    "field": "city",
                    "kind": "extra",
                    "label": "Город",
                    "value": "Москва",
                },
            ],
        )

        self.assertEqual(updated["approved_display_name"], "Иван")
        self.assertEqual(updated["display_name"], "Иван")
        self.assertEqual(updated["extra_fields"]["city"], "Москва")
        self.assertEqual(updated["profile_confidence"], 1.0)

    def test_delete_profile_field_clears_fixed_and_extra_values(self):
        profile = {
            "approved_display_name": "Иван",
            "display_name": "Иван",
            "extra_fields": {"city": "Москва", "occupation": "backend"},
        }

        without_name = delete_profile_field(profile, "approved_display_name")
        self.assertIsNone(without_name["approved_display_name"])
        self.assertIsNone(without_name["display_name"])

        without_city = delete_profile_field(without_name, "city")
        self.assertNotIn("city", without_city["extra_fields"])
        self.assertEqual(without_city["extra_fields"]["occupation"], "backend")

    def test_format_profile_view_renders_extra_fields(self):
        text = format_profile_view(
            {
                "approved_display_name": "Иван",
                "gender": "мужской",
                "interests": "музыка, код",
                "extra_fields": {"city": "Москва", "occupation": "backend"},
            }
        )

        self.assertIn("Имя: Иван", text)
        self.assertIn("Пол: мужской", text)
        self.assertIn("Интересы: музыка, код", text)
        self.assertIn("Город: Москва", text)
        self.assertIn("Занятие: backend", text)


if __name__ == "__main__":
    unittest.main()
