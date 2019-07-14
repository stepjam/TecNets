from data.dataset import Dataset


class MilSimPush(Dataset):

    def __init__(self, training_size=693, validation_size=76):
        super().__init__(name='mil_sim_push', img_shape=(125, 125, 3),
                         state_size=20, action_size=7, time_horizon=100,
                         training_size=training_size,
                         validation_size=validation_size)
