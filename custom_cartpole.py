import gymnasium as gym

class CenterAwareCartPole(gym.Wrapper):

    def step(self, action):

        obs, reward, terminated, truncated, info = self.env.step(action)

        cart_position = obs[0]
        pole_angle = obs[2]

        center_penalty = 0.1 * (cart_position ** 2)
        angle_penalty = 1.0 * abs(pole_angle)

        center_bonus = 0.5 * (1 - abs(cart_position) / 1.5)

        reward = (
            1
            - center_penalty
            - angle_penalty
            + center_bonus
        )

        if abs(cart_position) > 1.5:
            reward -= 50
            terminated = True

        return obs, reward, terminated, truncated, info