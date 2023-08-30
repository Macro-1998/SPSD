import os

#epoch_range = (19, 29)
exp_dir = 'exps/paper/final/gamma10_lambda30_weight0-9'

log_path = os.path.join(exp_dir, 'log.txt')
with open(log_path, 'r') as f:
    raw_lines = f.readlines()

    t2v_metrics = {
        "R1": [],
        "R5": [],
        "R10": [],
        "R50": [],
        "MedR": [],
        "MeanR": [],
        "geometric_mean_R1-R5-R10": []
    }
    v2t_metrics = {
        "R1": [],
        "R5": [],
        "R10": [],
        "R50": [],
        "MedR": [],
        "MeanR": [],
        "geometric_mean_R1-R5-R10": []
    } 
    overall_metrics = {
        "rsum": []
    }

    for line in raw_lines:
        log_info = line.lstrip().rstrip('\n')
        if "MSRVTT_jsfusion_test/t2v_metrics" in log_info or "ActivityNet_val1_test/t2v_metrics" in log_info or "LSMDC_full_test/t2v_metrics" in log_info:
            tag, score = log_info.split(': ')
            tag = tag.split('/')[-1]
            score = float(score)
            t2v_metrics[tag].append(score)
        elif "MSRVTT_jsfusion_test/v2t_metrics" in log_info or "ActivityNet_val1_test/v2t_metrics" in log_info or "LSMDC_full_test/v2t_metrics" in log_info:
            tag, score = log_info.split(': ')
            tag = tag.split('/')[-1]
            score = float(score)
            v2t_metrics[tag].append(score)
        elif "MSRVTT_jsfusion_test/overall_metrics/rsum" in log_info or "ActivityNet_val1_test/overall_metrics/rsum" in log_info or "LSMDC_full_test/overall_metrics/rsum" in log_info:
            tag, score = log_info.split(': ')
            tag = tag.split('/')[-1]
            score = float(score)
            overall_metrics[tag].append(score)
    
    best_rsum = max(overall_metrics['rsum'])
    best_epoch = overall_metrics['rsum'].index(best_rsum)
    print(os.path.basename(exp_dir) + '_best {}:'.format(best_epoch))
    for key in t2v_metrics:
        print('\tt2v_metrics/{}: {:.2f}'.format(key, t2v_metrics[key][best_epoch]))
    for key in v2t_metrics:
        print('\tv2t_metrics/{}: {:.2f}'.format(key, v2t_metrics[key][best_epoch]))
    for key in overall_metrics:
        print('\toverall_metrics/{}: {:.2f}'.format(key, overall_metrics[key][best_epoch]))
    
    start = max(best_epoch - 2, 0)
    end = min(best_epoch + 2 + 1, len(t2v_metrics['R1']))
    print(os.path.basename(exp_dir) + '_mean:')
    #start, end = epoch_range[0], epoch_range[1]
    for key in t2v_metrics:
        print('\tt2v_metrics/{}: {:.3f}'.format(key, sum(t2v_metrics[key][start:end]) / (end-start)))
    for key in v2t_metrics:
        print('\tv2t_metrics/{}: {:.3f}'.format(key, sum(v2t_metrics[key][start:end]) / (end-start)))
    for key in overall_metrics:
        print('\toverall_metrics/{}: {:.3f}'.format(key, sum(overall_metrics[key][start:end]) / (end-start)))
    