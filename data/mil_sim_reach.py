from data.dataset import Dataset


class MilSimReach(Dataset):

    def __init__(self, training_size=1500, validation_size=150):
        super().__init__(name='mil_sim_reach', img_shape=(64, 80, 3),
                         state_size=10, action_size=2, time_horizon=50,
                         training_size=training_size,
                         validation_size=validation_size)
