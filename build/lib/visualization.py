
import napari



def viz_labels(image,list_of_labels,list_of_names):
    with napari.gui_qt():

        view = napari.Viewer(ndisplay=2)
        # view.add_image(image)
        view.add_image(image)
        for i in range(len(list_of_labels)):
            # labels_layer = view.add_labels(old_labels-10)

            labels_layer = view.add_labels(list_of_labels[i], name=list_of_names[i])
    return None


