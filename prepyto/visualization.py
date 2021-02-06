
import napari
import numpy as np



def viz_labels(image,list_of_labels,list_of_names):
    with napari.gui_qt():

        view = napari.Viewer(ndisplay=2)
        # view.add_image(image)
        view.add_image(image)
        for i in range(len(list_of_labels)):
            # labels_layer = view.add_labels(old_labels-10)

            labels_layer = view.add_labels(list_of_labels[i], name=list_of_names[i])
    return None

def add_points_remove_labels(image, labels_to_analyze, additional_labels=None ):
    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(image)
        labels_layer = viewer.add_labels(labels_to_analyze)
        if additional_labels is not None:
            viewer.add_labels(additional_labels)
        points_to_remove_layer = viewer.add_points(np.empty((0,3)),name="spheres to remove")
        points_to_add_layer = viewer.add_points(np.empty((0,3)), name="spheres to add")

    return points_to_remove_layer.data, points_to_add_layer.data

def display_spheres(image, sphere_df):
    with napari.gui_qt():
        viewer = napari.view_image(image)
        points = viewer.add_points(np.empty((0,3)))
        points.add(sphere_df.center.to_list())
        points.size = (sphere_df.radius).to_list()
