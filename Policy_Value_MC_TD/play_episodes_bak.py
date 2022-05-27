##Play episodes function backup 
    def play_episodes(self, n_episodes, policy): # Play any number of episodes based on a policy
        track = []
        total_reward = 0
        for episode in range(n_episodes):
            terminated = False
            state = self.reset()

            while not terminated:
                action = self.get_best_choice(policy[state], key_return=True) # Select best action to perform in a current state
                next_state, reward, terminated = self.step(action) # Perform an action and observe environment
                total_reward += reward # Summarize total reward
                state = next_state # Update current state
            
            track.append(total_reward / (episode+1)) # Append current episode average reward to the tracking data structure
            
        average_reward = total_reward / n_episodes
        return total_reward, average_reward, track
