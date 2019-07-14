from unittest import main, TestCase
from data.mil_sim_push import MilSimPush
from data.mil_sim_reach import MilSimReach


class TestDatasets(TestCase):

    def test_mil_sim_reach_train(self):
        data = MilSimReach(training_size=10, validation_size=5)
        train, validation = data.training_set()
        self.assertEqual(len(train), 10)
        self.assertEqual(len(validation), 5)
        self.assertEqual(len(train[0]), 9)
        self.assertEqual(len(validation[0]), 5)
        self.assertIn('actions', train[0][0])
        self.assertIn('states', train[0][0])
        self.assertIn('image_files', train[0][0])

    def test_mil_sim_reach_test(self):
        data = MilSimReach(training_size=10, validation_size=10)
        test = data.test_set()
        self.assertEqual(len(test), 150)

    def test_mil_sim_push_train(self):
        data = MilSimPush(training_size=10, validation_size=5)
        train, validation = data.training_set()
        self.assertEqual(len(train), 10)
        self.assertEqual(len(validation), 5)
        self.assertEqual(len(train[0]), 12)
        self.assertEqual(len(validation[0]), 12)
        self.assertIn('actions', train[0][0])
        self.assertIn('states', train[0][0])
        self.assertIn('image_files', train[0][0])

    def test_mil_sim_push_test(self):
        data = MilSimPush(training_size=10, validation_size=10)
        test = data.test_set()
        self.assertEqual(len(test), 74)


if __name__ == '__main__':
    main()
