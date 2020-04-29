import numpy as np
import gym

from babyai.lsh import LSHBox

from babyai import lsh2

def make_lsh_struct(proj_mode):
    lsh_box = lsh2.lsh.LSHBox(lsh2.templates.EVERYTHING + lsh2.templates.STARTERS, allow_none=True)
    lsh_box_templates = lsh2.lsh.LSHBox(lsh2.templates.TEMPLATES.values(), allow_none=False)
    return (lsh_box, lsh_box_templates)


def lsh_query(lsh_struct, sentence):
    all_trees = set()
    subtrees = lsh2.blackbox_structured.get_subtrees(sentence)
    for tree in subtrees:
        all_trees.add(str(tree))
    proj = lsh2.blackbox_structured.project_one(lsh_struct[0], lsh_struct[1], sentence, subtrees)
    if len(proj) > 0:
        return proj[0].strip()
    else:
        return sentence


def evaluate_fixed_seeds(agent, env, episodes, seeds, orig_missions, alt_missions=None, proj_mode=None, proj_sentences=None):
    agent.model.eval()
    print(proj_mode)
    if proj_mode is not None:
        #lsh = LSHBox(proj_mode, proj_sentences)
        lsh_struct = make_lsh_struct(proj_mode)
    logs = {"num_frames_per_episode": [], "return_per_episode": [], "observations_per_episode": []}
    for i in range(len(seeds)):
        seed = seeds[i]
        env.seed(seed)
        obs = env.reset()

        print("---")
        print(obs["mission"])
        if obs["mission"] != orig_missions[i]:
            print(f"WARNING: failed to reproduce original instruction for {seed}; skipping")
            continue
        if alt_missions is None:
            mission = obs["mission"]
        else:
            mission = alt_missions[i]


        print(mission)
        if proj_mode is not None:
            #mission = lsh.query(mission)
            mission = lsh_query(lsh_struct, mission)
            print("->", mission)

        obs["mission"] = mission
        agent.on_reset()
        done = False

        num_frames = 0
        returnn = 0
        obss = []
        while not done:
            action = agent.act(obs)['action']
            obss.append(obs)
            obs, reward, done, _ = env.step(action)
            obs["mission"] = mission
            agent.analyze_feedback(reward, done)
            num_frames += 1
            returnn += reward
        print("score", returnn)
        logs["observations_per_episode"].append(obss)
        logs["num_frames_per_episode"].append(num_frames)
        logs["return_per_episode"].append(returnn)
    return logs


# Returns the performance of the agent on the environment for a particular number of episodes.
def evaluate(agent, env, episodes, model_agent=True, offsets=None):
    # Initialize logs
    if model_agent:
        agent.model.eval()
    logs = {"num_frames_per_episode": [], "return_per_episode": [], "observations_per_episode": []}

    if offsets:
        count = 0

    for i in range(episodes):
        if offsets:
            # Ensuring test on seed offsets that generated successful demonstrations
            while count != offsets[i]:
                obs = env.reset()
                count += 1

        obs = env.reset()
        agent.on_reset()
        done = False

        num_frames = 0
        returnn = 0
        obss = []
        while not done:
            action = agent.act(obs)['action']
            obss.append(obs)
            obs, reward, done, _ = env.step(action)
            agent.analyze_feedback(reward, done)
            num_frames += 1
            returnn += reward


        logs["observations_per_episode"].append(obss)
        logs["num_frames_per_episode"].append(num_frames)
        logs["return_per_episode"].append(returnn)
    if model_agent:
        agent.model.train()
    return logs


def evaluate_demo_agent(agent, episodes):
    logs = {"num_frames_per_episode": [], "return_per_episode": []}

    number_of_demos = len(agent.demos)

    for demo_id in range(min(number_of_demos, episodes)):
        logs["num_frames_per_episode"].append(len(agent.demos[demo_id]))

    return logs


class ManyEnvs(gym.Env):

    def __init__(self, envs):
        self.envs = envs
        self.done = [False] * len(self.envs)

    def seed(self, seeds):
        [env.seed(seed) for seed, env in zip(seeds, self.envs)]

    def reset(self):
        many_obs = [env.reset() for env in self.envs]
        self.done = [False] * len(self.envs)
        return many_obs

    def step(self, actions):
        self.results = [env.step(action) if not done else self.last_results[i]
                        for i, (env, action, done)
                        in enumerate(zip(self.envs, actions, self.done))]
        self.done = [result[2] for result in self.results]
        self.last_results = self.results
        return zip(*self.results)

    def render(self):
        raise NotImplementedError


# Returns the performance of the agent on the environment for a particular number of episodes.
def batch_evaluate(agent, env_name, seed, episodes, return_obss_actions=False):
    num_envs = min(256, episodes)

    envs = []
    for i in range(num_envs):
        env = gym.make(env_name)
        envs.append(env)
    env = ManyEnvs(envs)

    logs = {
        "num_frames_per_episode": [],
        "return_per_episode": [],
        "observations_per_episode": [],
        "actions_per_episode": [],
        "seed_per_episode": []
    }

    for i in range((episodes + num_envs - 1) // num_envs):
        seeds = range(seed + i * num_envs, seed + (i + 1) * num_envs)
        env.seed(seeds)

        many_obs = env.reset()

        cur_num_frames = 0
        num_frames = np.zeros((num_envs,), dtype='int64')
        returns = np.zeros((num_envs,))
        already_done = np.zeros((num_envs,), dtype='bool')
        if return_obss_actions:
            obss = [[] for _ in range(num_envs)]
            actions = [[] for _ in range(num_envs)]
        while (num_frames == 0).any():
            action = agent.act_batch(many_obs)['action']
            if return_obss_actions:
                for i in range(num_envs):
                    if not already_done[i]:
                        obss[i].append(many_obs[i])
                        actions[i].append(action[i].item())
            many_obs, reward, done, _ = env.step(action)
            agent.analyze_feedback(reward, done)
            done = np.array(done)
            just_done = done & (~already_done)
            returns += reward * just_done
            cur_num_frames += 1
            num_frames[just_done] = cur_num_frames
            already_done[done] = True

        logs["num_frames_per_episode"].extend(list(num_frames))
        logs["return_per_episode"].extend(list(returns))
        logs["seed_per_episode"].extend(list(seeds))
        if return_obss_actions:
            logs["observations_per_episode"].extend(obss)
            logs["actions_per_episode"].extend(actions)

    return logs
