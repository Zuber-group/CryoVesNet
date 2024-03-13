
import napari
import numpy as np
import prepyto


def viz_labels(image,list_of_labels,list_of_names):
    with napari.gui_qt():

        view = napari.Viewer(ndisplay=2)
        # view.add_image(image)
        view.add_image(image)
        for i in range(len(list_of_labels)):
            # labels_layer = view.add_labels(old_labels-10)

            labels_layer = view.add_labels(list_of_labels[i], name=list_of_names[i])
    return None

def add_points_remove_labels(pipe, labels_to_analyze, additional_labels=None ):
    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(pipe.image)
        viewer.add_labels(labels_to_analyze)

        @viewer.bind_key('s')
        # Press 's' to save the labels
        def print_message(viewer):
            print('Save procedures')
            yield
            minimum_box_size = pipe.voxel_size * 50
            pipe.mancorr_labels = prepyto.remove_labels_under_points(labels_to_analyze, points_to_remove_layer.data)
            pipe.mancorr_labels = prepyto.add_sphere_labels_under_points(pipe.image, pipe.mancorr_labels, points_to_add_layer.data,
                                                                         points_to_add_layer.size[:,1], minimum_box_size)
        if additional_labels is not None:
            viewer.add_labels(additional_labels)
        points_to_remove_layer = viewer.add_points(data= np.empty((0, 3)), ndim=3, n_dimensional=True, name="spheres to remove")
        points_to_add_layer = viewer.add_points(data=np.empty((0, 3)), ndim=3, size=[], opacity=0.3,
                                                n_dimensional=True, name="spheres to add")
    points_to_remove = points_to_remove_layer.data
    points_to_add = points_to_add_layer.data
    points_to_add_sizes = points_to_add_layer.size[:,1]
    return points_to_remove, points_to_add, points_to_add_sizes

def display_spheres(image, sphere_df):
    with napari.gui_qt():
        viewer = napari.view_image(image)
        points = viewer.add_points(np.empty((0,3)))
        points.add(sphere_df.center.to_list())
        points.size = (sphere_df.radius).to_list()
