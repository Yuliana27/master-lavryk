from termcolor import colored

def early_stopping(epoch, epoch_val_loss, previous_loss, count_paitence, patience):
    if epoch_val_loss - previous_loss>0.2:
        count_paitence += 1
        print(colored(f'Epoch {epoch} validation set loss is greater than epoch {epoch -1} loss.','blue'))
        print(colored(f'Paitence Counter value: {count_paitence}', 'green'))
        if count_paitence >= patience:
            print(colored('Stopping early as validation score has stopped improving','magenta', attrs=['bold']))
            return ['stop']
    else:
        count_paitence = 0
    previous_loss = epoch_val_loss
    
    return [previous_loss, count_paitence]