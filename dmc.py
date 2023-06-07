import gym
import numpy as np


class DMC(gym.core.Env):
    def __init__(self, name, action_repeat=1, size=(64, 64), camera=None):
        domain, task = name.split("-", 1)
        if domain.startswith('manip'):
            from dm_control import manipulation
            self._env = manipulation.load(task)
        else:
            assert task is None
            self._env = domain()
        self._action_repeat = action_repeat
        self._size = size
        if camera is None:
            camera = dict(quadruped=2).get(domain, 0)
        self._camera = camera
        self.action_bins = np.linspace(self.action_space.low/2, self.action_space.high/2, 7).transpose(1, 0)

    def random_action(self):
        return np.random.randint(0, 7, 9)

    @property
    def observation_space(self):
        spaces = {}
        for key, value in self._env.observation_spec().items():
            v_shape = value.shape
            if len(value.shape) < 3:
                key = 'vec' + key
                if len(value.shape) == 0:
                    v_shape = (1,)
            spaces[key] = gym.spaces.Box(-np.inf, np.inf, v_shape, dtype=np.float32)
        spaces["image"] = gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        spec = self._env.action_spec()
        return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

    def step(self, action):
        action = [x[i] for x, i in zip(self.action_bins, action)]
        time_step = self._env.step(action)
        reward = time_step.reward
        obs_ = dict(time_step.observation)
        obs = {}
        for k, v in obs_.items():
            if len(v.shape) < 3:
                if len(v.shape) == 0:
                    v = np.array([v])
                obs['vec' + k] = v
        # obs["image"] = self.render()
        # There is no terminal state in DMC
        obs["is_terminal"] = False if time_step.first() else time_step.discount == 0
        obs["is_first"] = time_step.first()
        done = time_step.last()
        trunc = not done and time_step.discount == 0
        info = {"discount": np.array(time_step.discount, np.float32)}
        vs = []
        for k, v in obs.items():
            if k.startswith('vec'):
                vs.append(v)
        vs = np.concatenate(vs, axis=-1)
        return vs, reward, done, trunc, info

    def reset(self):
        time_step = self._env.reset()
        obs_ = dict(time_step.observation)
        obs = {}
        for k, v in obs_.items():
            if len(v.shape) < 3:
                if len(v.shape) == 0:
                    v = np.array([v])
                obs['vec' + k] = v
        # obs["image"] = self.render()
        obs["is_terminal"] = False if time_step.first() else time_step.discount == 0
        obs["is_first"] = time_step.first()
        vs = []
        for k, v in obs.items():
            if k.startswith('vec'):
                vs.append(v)
        vs = np.concatenate(vs, axis=-1)
        return vs, {}

    def render(self, *args, **kwargs):
        if kwargs.get("mode", "rgb_array") != "rgb_array":
            raise ValueError("Only render mode 'rgb_array' is supported.")
        return self._env.physics.render(*self._size, camera_id=self._camera)
    

if __name__ == '__main__':
    env = DMC('manip-reach_site_features')
    obs, _ = env.reset()
    action = np.random.randint(0, 7, 9)
    next_obs, reward, done, _, _ = env.step(action)
    print(next_obs)
