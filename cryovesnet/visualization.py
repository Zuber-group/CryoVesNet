
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


import napari
from magicgui import magicgui
import numpy as np
import cryovesnet  # Ensure your module is correctly imported

def add_points_modify_labels(pipe, max_diameter=50):
    viewer = napari.Viewer()
    viewer.add_image(pipe.image)

    # Single labels layer
    labels_layer = viewer.add_labels(getattr(pipe, pipe.last_output_array_name), name="Editable Labels")

    # Single points layer for modification operations
    points_layer = viewer.add_points(np.empty((0, 3)), ndim=3, size=10, opacity=0.5, n_dimensional=True, name="Modification Points")

    # Backup for undo functionality
    labels_backup = None

    def backup_labels():
        nonlocal labels_backup
        labels_backup = np.copy(labels_layer.data)

    @magicgui(call_button="Compute Labels [C]")
    def compute_labels():
        backup_labels()
        if len(points_layer.data) > 0:
            modified_labels = cryovesnet.add_sphere_labels_under_points(
                pipe.image, labels_layer.data, points_layer.data,
                points_layer.size[:] if points_layer.size.size > 0 else [],  max_diameter//pipe.voxel_size )
            labels_layer.data = modified_labels
            print("Labels added.")
        # Clear points after applying changes
        points_layer.selected_data = set(range(len(points_layer.data)))
        points_layer.remove_selected()

    @magicgui(call_button="Remove Labels [R]")
    def remove_labels():
        backup_labels()
        if len(points_layer.data) > 0:
            modified_labels = cryovesnet.remove_labels_under_points(labels_layer.data, points_layer.data)

            labels_layer.data = modified_labels
            print("Labels removed.")
        # Clear points after applying changes
        points_layer.selected_data = set(range(len(points_layer.data)))
        points_layer.remove_selected()

    @magicgui(call_button="Add Spherical Labels [P]")
    def add_spherical_labels():
        backup_labels()
        if len(points_layer.data) > 0:
            max_label = labels_layer.data.max()
            for i, point in enumerate(points_layer.data):
                size = points_layer.size[i] if points_layer.size.size > 0 else 10
                radius = int(np.round(size / 2))
                label_value = i + max_label + 1  # Unique label for each new sphere
                # label_value = 3000
                modified_labels = cryovesnet.put_spherical_label_in_array(
                    labels_layer.data,
                    np.round(point).astype(int),
                    radius,
                    label_value,
                    inplace=False
                )
                labels_layer.data = modified_labels
            print("Spherical labels added.")
        # Clear points after applying changes
        points_layer.selected_data = set(range(len(points_layer.data)))
        points_layer.remove_selected()

    @magicgui(call_button="Undo [U]")
    def undo():
        if labels_backup is not None:
            labels_layer.data = labels_backup
            print("Undo last change.")
        else:
            print("No changes to undo.")

    @magicgui(call_button="Save [S]")
    def save():
        if labels_layer.data is not None:
            print('Saving changes...')
            cryovesnet.save_label_to_mrc(labels_layer.data, pipe.mancorr_labels_path, template_path=pipe.image_path)
            pipe.last_output_array_name = 'mancorr_labels'  # Update reference to reflect saved state
            print("Changes saved in ",pipe.last_output_array_name)
        else:
            print("No changes to save.")

    @magicgui(call_button="Exit")
    def exit_without_save():
        from qtpy.QtCore import QTimer
        QTimer.singleShot(100, viewer.close)  # Delay closing by 100 ms

    def compute_labels_b(event=None):
        compute_labels()
    def remove_labels_b(event=None):
        remove_labels()
    def add_spherical_labels_b(event=None):
        add_spherical_labels()
    def save_b(event=None):
        save()

    def undo_b(event=None):
        undo()

    viewer.bind_key('C', compute_labels_b)
    viewer.bind_key('R', remove_labels_b)
    viewer.bind_key('P', add_spherical_labels_b)
    viewer.bind_key('S', save_b)
    viewer.bind_key('U', undo_b)

    viewer.window.add_dock_widget(compute_labels, area='right')
    viewer.window.add_dock_widget(remove_labels, area='right')
    viewer.window.add_dock_widget(add_spherical_labels, area='right')
    viewer.window.add_dock_widget(save, area='right')
    viewer.window.add_dock_widget(undo, area='right')
    viewer.window.add_dock_widget(exit_without_save, area='right')

    napari.run()





def add_points_remove_labels_v1(pipe, max_diameter=50):
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
        modified_labels = cryovesnet.add_sphere_labels_under_points(pipe.image, modified_labels, points_to_add, points_to_add_sizes,  max_diameter//pipe.voxel_size)

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
