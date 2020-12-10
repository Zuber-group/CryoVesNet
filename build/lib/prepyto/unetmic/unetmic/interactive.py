import numpy as np

def get_label(layer, event):
    cords = np.round(layer.coordinates).astype(int)
    val = layer.get_value()
    if val is None:
        return
    if val != 0:
        if 'Shift' in event.modifiers:
            layer.selected_label = val
        elif layer.mode =='pan_zoom':
            data = layer.data
            data[data == val] = 0
            layer.data = data

        msg = f'clicked at {cords} on blob {val} which is erased'
    else:
        msg = f'clicked at {cords} on background which is ignored'
    layer.status = msg
    print(msg)
   