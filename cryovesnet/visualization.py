
import napari
import numpy as np
import cryovesnet
from napari.settings import get_settings
from magicgui import magicgui

get_settings().application.ipy_interactive = False

def viz_labels(image,list_of_labels,list_of_names):
    with napari.gui_qt():

        view = napari.Viewer(ndisplay=2)
        # view.add_image(image)
        view.add_image(image)
        for i in range(len(list_of_labels)):
            # labels_layer = view.add_labels(old_labels-10)

            labels_layer = view.add_labels(list_of_labels[i], name=list_of_names[i])
    return None


def add_points_remove_labels(pipe):
    viewer = napari.Viewer()
    viewer.add_image(pipe.image)

    labels_layer = viewer.add_labels(getattr(pipe, pipe.last_output_array_name), name="Editable Labels")

    points_to_remove_layer = viewer.add_points(np.empty((0, 3)), ndim=3, n_dimensional=True, name="Points to Remove")
    points_to_add_layer = viewer.add_points(np.empty((0, 3)), ndim=3, size=None, opacity=0.3, n_dimensional=True, name="Points to Add")

    @magicgui(call_button="Apply Changes [R]")
    def apply_changes():
        points_to_remove = points_to_remove_layer.data
        points_to_add = points_to_add_layer.data
        points_to_add_sizes = points_to_add_layer.size[:] if points_to_add_layer.size.size > 0 else []

        # Use the current labels_layer.data for updates to ensure persistence
        modified_labels = cryovesnet.remove_labels_under_points(labels_layer.data, points_to_remove)
        modified_labels = cryovesnet.add_sphere_labels_under_points(pipe.image, modified_labels, points_to_add, points_to_add_sizes, pipe.voxel_size * 50)

        # Directly update the labels layer with the new modifications
        labels_layer.data = modified_labels
        print("Changes applied, not yet saved.")

        # Reset the points layers
        points_to_remove_layer.selected_data = set(range(len(points_to_remove_layer.data)))
        # points_to_remove_layer.size = np.empty((0, 1))
        points_to_add_layer.selected_data = set(range(len(points_to_add_layer.data)))
        points_to_add_layer.remove_selected()
        points_to_remove_layer.remove_selected()

    @magicgui(call_button="Save [S]")
    def save():
        if labels_layer.data is not None:
            print('Saving changes...')
            cryovesnet.save_label_to_mrc(labels_layer.data, pipe.mancorr_labels_path, template_path=pipe.image_path)
            pipe.last_output_array_name = 'mancorr_labels'  # Update reference to reflect saved state
        else:
            print("No changes to save.")

    @magicgui(call_button="Exit")
    def exit_without_save():
        # save()  # Perform save operation
        from qtpy.QtCore import QTimer
        QTimer.singleShot(100, viewer.close)  # Delay closing by 100 ms

    def apply_changes_b(event=None):
        apply_changes()
    def save_b(event=None):
        save()

    viewer.bind_key('R', apply_changes_b)
    viewer.bind_key('S', save_b)

    viewer.window.add_dock_widget(apply_changes, area='right')
    viewer.window.add_dock_widget(save, area='right')
    viewer.window.add_dock_widget(exit_without_save, area='right')

    napari.run()



def display_spheres(image, sphere_df):
    with napari.gui_qt():
        viewer = napari.view_image(image)
        points = viewer.add_points(np.empty((0,3)))
        points.add(sphere_df.center.to_list())
        points.size = (sphere_df.radius).to_list()
