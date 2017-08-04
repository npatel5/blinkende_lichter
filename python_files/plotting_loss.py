import matplotlib.pyplot as plt

def plot_average_loss(training_loss, validation_loss):
    avg_ep = [training_loss[x:x+2] for x in range(0, len(training_loss), 2)]
    avg_val_ep = [validation_loss[x:x+4] for x in range(0, len(validation_loss), 4)]

    for i in range(len(avg_ep)):
        avg_ep[i]=sum(avg_ep[i])/len(avg_ep[i])

    for i in range(len(avg_val_ep)):
        avg_val_ep[i]=sum(avg_val_ep[i])/len(avg_val_ep[i])
    
    fig = plt.figure(figsize=(20,20))
    plt.plot(avg_val_ep)
    plt.plot(avg_ep)
    plt.show()