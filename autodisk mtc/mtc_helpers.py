from IPython.display import clear_output
import time

def check_time(func):
    def inner(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Time taken to execute function [{func}] is ", end-start)
        return result
    return inner

def test(data):
    print(type(data))
    try: print(data.shape)
    except: print(f'{data} does not have a .shape property')
    
def timeToFinish(imgh, imgw, avg_time, current_time):
    #determine percent completed
    
    num_patterns = imgh*imgw
    current_pattern = avg_time[1]+1
    percent_complete = current_pattern/num_patterns
    percent_complete = round(percent_complete,4)
    
    #estimate time
    tot_time_est = current_time/percent_complete
    time_remaining = tot_time_est-current_time
    
    avg_time = [round(((avg_time[0]*avg_time[1]) + tot_time_est)/(current_pattern),2),current_pattern]
    print(f'Percent completed: {percent_complete*100}%',flush=True)
    print(f'Estimated time to finish: {round(time_remaining,2)}',flush=True)
    print(f'Total estimated time for task: {tot_time_est}',flush=True)
    print(f'Average total time to finish: {avg_time[0]}, [{avg_time[1]}/{num_patterns}]',flush=True)
    print(f'Current time = {round(current_time,2)}',flush=True)
    clear_output(wait=True)
    return avg_time 