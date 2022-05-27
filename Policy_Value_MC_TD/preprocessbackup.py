states = ['A', 'B', 'C', 'D']
def get_env_vars(rewards):
    Predirs = {}
    Pjams = {}
    RewJams = {}
    RewNorms = {}
    
    for x in rewards.keys():
        num = rewards[x]
        splitter = x.split('{')[1].split('}')[0]
        temp = [splitter[0], splitter[2]]
        
        if('1' in x):
            temp.append(1)
        elif('2' in x):
            temp.append(2)
        temp = tuple(temp)
        
        if('State' in x):
            Predirs[temp] = num
        elif('Pjam' in x):
            Pjams[temp] = num
        elif('Reward_jam' in x):
            RewJams[temp] = num
        elif('Reward_normal' in x):
            RewNorms[temp] = num
    
    return Predirs, Pjams, RewJams, RewNorms 
