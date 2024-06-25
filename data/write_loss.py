
def write_loss(train_loss_list,dev_loss_list,loss_list):
    with open('saved_models/00/loss.txt', 'w') as file:
        file.write("train_loss: ")
        file.write(str(train_loss_list))
        file.write('\n\n')
        file.write("dev_loss: ")
        file.write(str(dev_loss_list))
        file.write('\n\n')
        file.write("loss: ")
        file.write(str(loss_list))

