import numpy as np

alpha = 0.01  # taux d'apprentisssage
agent = [1, 0]  # position initiale [ligne, colonne]
theta = np.array([4.0, 4.0, 4.0, 4.0])  # parametres pour les 4 actions

def softmax(t):
    exp_t = np.exp(t - np.max(t))  # pour la stabilité numérique
    return exp_t / np.sum(exp_t)

def choose_action(theta):
    probs = softmax(theta)
    return np.random.choice(len(probs), p=probs)

def gradient_log_policy(action, theta):
    probs = softmax(theta)
    grad = -probs
    grad[action] += 1
    return grad

def move(agent, action):
    x, y = agent
    if action == 0:  # gauche
        y = max(y - 1, 0)
    elif action == 1:  # droite
        y = min(y + 1, 1)
    elif action == 2:  # bas
        x = min(x + 1, 1)
    elif action == 3:  # haut
        x = max(x - 1, 0)
    return [x, y]

def get_state(agent):
    return 0 if agent == [1, 1] else 1  # état final = [1,1]

def get_reward(agent, action):
    if agent == [1, 1] and action == 1:  # action droite au bon endroit
        return 10
    else:
        return -1

# Boucle d'apprentissage
for episode in range(10000):
    agent = [1, 0]
    actions = []
    rewards = []

    for step in range(100):
        state = get_state(agent)
        if state == 0:  # atteint le but
            break
        
        action = choose_action(theta)
        actions.append(action)
        
        agent = move(agent, action)
        reward = 10 if agent == [1, 1] and action == 1 else -1
        rewards.append(reward)
        
        if reward == 10:
            break

    # Calcul des retours G
    G = []
    for t in range(len(rewards)):
        Gt = sum(rewards[t:])  # somme des récompenses futures
        G.append(Gt)

    # Mise à jour des parametres
    for t in range(len(actions)):
        theta += alpha * G[t] * gradient_log_policy(actions[t], theta)

print("Probabilités finales des actions :", softmax(theta))
print("la meilleur action :", np.argmax(softmax(theta)))
