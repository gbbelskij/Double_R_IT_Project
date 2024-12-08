import numpy as np


class Course:
    FRONT = 1
    BACK = 2
    ML = 3
    SQL = 4


# Dictionary from number to course name
courses = {
    1: 'Frontend',
    2: 'Backend',
    3: 'ML',
    4: 'SQL'
}


def f(x):
    return 1 / (1 + np.exp(-x))


# Create first layer of sinapses: 8 hiden layers + 4 input layers
W1 = np.array([[-0.234, 0.123, 0.012, -0.189],
               [0.091, 0.278, -0.012, 0.234],
               [-0.189, 0.012, 0.234, -0.091],
               [0.012, 0.189, -0.278, 0.012],
               [-0.091, 0.234, -0.012, 0.189],
               [0.278, -0.012, 0.012, 0.234],
               [-0.012, 0.012, 0.189, -0.234],
               [0.234, -0.189, 0.012, 0.091],])

# Create second layer of sinapses: 4 output layers + 8 hiden layers
W2 = np.array([[-0.012, 0.234, 0.012, -0.189, 0.091, 0.278, -0.012, 0.012],
               [0.189, -0.012, 0.234, -0.091, 0.012, 0.012, 0.278, -0.234],
               [0.012, 0.012, 0.012, 0.234, -0.189, 0.091, 0.278, -0.012],
               [-0.234, 0.091, 0.012, 0.012, 0.278, -0.012, 0.012, 0.189],])

# Func go through network, return vector of output values from hiden layer (hidden_lay_out) and output layer (output_lay_out)


def go_through(inp):
    sum = np.dot(W1, inp)
    hidden_lay_out = np.array([f(x) for x in sum])

    sum = np.dot(W2, hidden_lay_out)
    output_lay_out = np.array([f(x) for x in sum])
    return (hidden_lay_out, output_lay_out)


def train(epoch, W1, W2):
    lmd = 0.01          # learning step
    N = 10000           # number of iterations during training
    count = len(epoch)  # number of examples
    for k in range(N):
        
        rand_example = epoch[np.random.randint(0, count)]               # take random example frome given
        hidden_lay_out, output_lay_out = go_through(rand_example[0:4])  # go through network and remember output
        expected_neyron_position = rand_example[-1] - 1                 # position of expected neyron in outpat layer (given in epoch)
        real_return_neyron_position = np.argmax(output_lay_out)         # real resulting neyron position
        e = 1 - output_lay_out[expected_neyron_position]                # count error
        delta = e * output_lay_out[expected_neyron_position] * \
            (1 - output_lay_out[expected_neyron_position])              # Within df(x) i take f(x) * (1 - f(x))

        # Correction weights in hiden-output layer (adding to expected neyron and subtract from real resulting neyron)
        for i, f1x in enumerate(hidden_lay_out):
            W2[expected_neyron_position][i] += lmd * delta * f1x
        for i, f1x in enumerate(hidden_lay_out):
            W2[real_return_neyron_position][i] -= lmd * delta * f1x

        # Counting delta in hiden layer
        delta21 = delta * W2[expected_neyron_position] * \
            hidden_lay_out * (1 - hidden_lay_out)
        delta22 = delta * W2[real_return_neyron_position] * \
            hidden_lay_out * (1 - hidden_lay_out)

        # Correction weights in input-hiden layer (also adding to expected and subtract from real resulting neyron)
        for i in range(len(W1)):
            W1[i, :] = W1[i, :] + \
                np.array(rand_example[0:4]) * delta21[i] * lmd
        for i in range(len(W1)):
            W1[i, :] = W1[i, :] - \
                np.array(rand_example[0:4]) * delta22[i] * lmd

    return W1, W2


# Examples made by GPT:
epoch = [
    # Фронтендеры (любят только фронтенд, но иногда и SQL)
    (1, 0, 1, 0, Course.FRONT),  # Любит HTML/CSS и Frontend
    (1, 0, 1, 1, Course.SQL),    # Любит HTML/CSS, Frontend, может попробовать SQL
    (1, 0, 1, 0, Course.FRONT),  # Повторяющийся правильный пример для обучения

    # Бэкендеры (любят только backend, но иногда и SQL)
    (0, 1, 0, 1, Course.BACK),   # Любит C++ и Backend
    (0, 1, 0, 1, Course.BACK),   # Любит C++ и Backend
    (0, 1, 0, 1, Course.SQL),    # Может попробовать SQL

    # Специалисты по машинному обучению (любят C++ и ML)
    (0, 1, 0, 0, Course.ML),     # Любит C++ и ничего больше
    (0, 1, 1, 0, Course.ML),     # Любит C++, иногда интересуется Frontend

    # SQL-специалисты (любят только SQL)
    (0, 0, 0, 1, Course.SQL),    # Любит только SQL
    (1, 0, 0, 1, Course.SQL),    # Любит HTML/CSS и SQL

    # Смешанные примеры (редкие)
    (1, 1, 1, 1, Course.SQL),    # Любит все и может выбрать SQL
    (1, 1, 1, 0, Course.FRONT),  # Многозадачник, но тяготеет к Frontend
]

train(epoch, W1, W2)    # start the training

for i, x in enumerate(epoch):
    first, final = go_through(x[0:4])           # go through network
    course = np.argmax(final)                   # resulting neyron position

    emoji = "✅" if (course + 1 == x[-1]) else "⚠️"

    print(f"Recomended course by AI: {courses[course + 1].ljust(10)}; "
          f"Expected: {courses[x[-1]].ljust(10)} {emoji}")
