import numpy as np
import napari
import mrcfile
from .unetmic.unetmic import interactive
import sys
from pathlib import Path



def query_yes_no(question, default="yes"):
    valid = {"yes": True, "y": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")




def interactive_cleaner(real_image,myimage_labels):
    # if sys.flags.interactive != 1 or not hasattr(QtCore, 'PYQT_VERSION'):
    #     pg.QtGui.QApplication.exec_()
    # %gui qt5
    with napari.gui_qt():
        view = napari.Viewer(ndisplay=2)
        view.add_image(real_image)
        # labels_layer = view.add_labels(myimage_labels , name='Fine')
        labels_layer = view.add_labels(myimage_labels)
        labels_layer.mouse_drag_callbacks=[interactive.get_label]
        # temp=view.layers[1].data
    yesOrNo=query_yes_no("Are you Done with interactive cleaning?")
    return myimage_labels




def mrc_header_cleaner(source_mrc_path,template_mrc_path, target_mrc_path):
    """
    creates a new mrc file from a label file with header modified to correspond to the one of the original dataset
    :param source_mrc_path: (path) the header of this file will be modified
    :param template_mrc_path: (path) the headr of this file serves as a template
    :param target_mrc_path: (path) the cleaned mrc file path
    :return:
    """
    #firstPart = os.path.split(source_mrc)[0] + '/'
    #name_of_the_file = os.path.splitext(os.path.split(source_mrc)[1])[0]+'_clean.mrc'
    new_target_labels = mrcfile.open(source_mrc_path).data.astype(np.uint16)
    with mrcfile.new(target_mrc_path, overwrite=True) as mrc:
        tomo = mrcfile.open(template_mrc_path, header_only=True)
        mrc.set_data(new_target_labels)
        mrc.voxel_size = tomo.voxel_size
        mrc.header['origin'] = tomo.header['origin']
    return



def ask_file_path(mypath,file_extension=('.mrc', '.nad', '.rec')):
    dataset_lst = [e for e in mypath.iterdir()]
    for i, p in enumerate(dataset_lst):
        print(f"{i}: {p.name}")
    while True:
        choice = input("Which mrc do you want to use as a reference for evaluation (give the number) ?")
        if choice.isdigit():
            if int(choice) in range(len(dataset_lst)):
                p = Path(dataset_lst[int(choice)])
                print(p)
                if p.exists() and p.suffix in file_extension:
                    break
        print("invalid answer, retry!")
    return p