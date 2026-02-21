import secrets
import glob
import os
from datetime import datetime
from .utils import save_dictionary

def save_model(parameters, model_params_group):
    print('Saving model...')
    id = secrets.token_hex(32)
    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if parameters['io_dict']['remove_old_model']:
        model_names = glob.glob(os.path.join(parameters['io_dict']['model_dir'], 'model_*'))
        if len(model_names) > 0:
            for model_name in model_names:
                os.remove(model_name)

    model_data = dict(
        model=parameters['model_dict']['model'],
        run_information=model_params_group,
        id=id,
        time=str(now),
        parameters=parameters,
        system_info=parameters['device_dict']['system_info']
    )
    print(os.path.join(parameters['io_dict']['model_dir'], 'model_' + str(id) + '_' + str(now)))
    save_dictionary(os.path.join(parameters['io_dict']['model_dir'], 'model_' + str(id) + '_' + str(now)),
                    model_data)










