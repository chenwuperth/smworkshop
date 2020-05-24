"""
inference script for both batch transform and endpoint
"""
import os

def input_fn(request_body, request_content_type):
    """Takes request data and de-serializes the data into an object for prediction.
        When an InvokeEndpoint operation is made against an Endpoint running SageMaker model server,
        the model server receives two pieces of information:
            - The request Content-Type, for example "application/json"
            - The request data, which is at most 5 MB (5 * 1024 * 1024 bytes) in size.
        The input_fn is responsible to take the request data and pre-process it before prediction.
    Args:
        input_data (obj): the request data.
        content_type (str): the request Content-Type.
    Returns:
        (obj): data ready for prediction.
    """

    #print(request_body)
    lines = request_body.split(os.linesep)
    arrs = []
    for line in lines:
        # we know it is CSV format
        line = line.strip()
        ds = line.split(',')[1:] # remove the first element
        if (len(ds) != 13):
            print(f'short line {line}')
            continue
        try:
            ds_arr = np.array([float(x) for x in ds])
        except:
            print(f'Fail to convert to float {line}')
            continue
        arrs.append(ds_arr)
    return np.stack(arrs)