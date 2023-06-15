from typing import Optional, Union, List, Dict, Callable
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
import supervisely.nn.inference.gui as GUI
import supervisely.app.widgets as Widgets
from supervisely.task.progress import Progress
from supervisely import Api

class MMDetectionGUI(GUI.InferenceGUI):
    def __init__(
        self,
        models: Union[List[Dict[str, str]], Dict[str, List[Dict[str, str]]]],
        api: Api,
        support_pretrained_models: Optional[bool],
        support_custom_models: Optional[bool],
        custom_model_link_type: Optional[Literal["file", "folder"]] = "file",
    ):
        self._all_models = models
        super(MMDetectionGUI, self).__init__(
            models=models["object detection"], 
            api=api,
            support_pretrained_models=support_pretrained_models,
            support_custom_models=support_custom_models,
            custom_model_link_type=custom_model_link_type,
        )

        self._task_infobox = Widgets.NotificationBox(
            title="INFO: How to select task?",
            description="At this step you should select deep learning problem that you want to solve. Outputs of object detection models - only bounding boxes with confidence. Outputs of instance segmentation models in addition contain object masks. Selected task at this step defines models list to choose. If you want to serve model based on already trained custom model, choose the appropriate task.",
            box_type='info',
        )
        self._select_task_radio = Widgets.RadioGroup(
            [
                Widgets.RadioGroup.Item(
                    value="object detection"
                ),
                Widgets.RadioGroup.Item(
                    value="instance segmentation"
                ),
            ]
        )
        self._select_task_radio.set_value("object detection")
        self._select_task_field = Widgets.Field(
            content=self._select_task_radio,
            title="Select deep learning problem to solve",
        )
        self._task_loading_text = Widgets.Text(
            text="Loading model configs of selected task...",
            status='text',
        )
        self._task_loading_text.hide()
        self._select_task_button = Widgets.Button(
            text="Select task"
        )

        self._reselect_task_button = Widgets.Button(
            text="Choose another task",
            button_type="warning",
            plain=True,
        )
        self._reselect_task_button.hide()

        self._task_card = Widgets.Card(
            title="MMDetection task",
            description="Select task from the list below",
            collapsable=True,
            content=Widgets.Container([
                self._task_infobox,
                self._select_task_field,
                self._select_task_button,
                self._task_loading_text,
                self._reselect_task_button,
            ], gap=5)
        )

        self._models_card = Widgets.Card(
            title="Select model",
            description="Choose model architecture and how weights should be initialized",
            collapsable=True,
            content=self._tabs,

        )
        self._models_card.collapse()
        self._models_card.lock()


        @self._select_task_button.click
        def reload_models():
            self._select_task_button.loading = True
            self._task_loading_text.show()
            task_type = self._select_task_radio.get_value()
            models = self._all_models[task_type]
            self._set_pretrained_models(models)
            self._select_task_button.loading = False
            self._select_task_button.hide()
            self._task_loading_text.hide()
            self._reselect_task_button.show()
            self._models_card.uncollapse()
            self._models_card.unlock()
            self._select_task_radio.disable()


        @self._reselect_task_button.click
        def reselect_task():
            self._select_task_radio.enable()
            self._reselect_task_button.hide()
            self._select_task_button.show()
            self._models_card.lock()
            self._models_card.collapse()

    def get_task_type(self) -> str:
        return self._select_task_radio.get_value()

    def change_model(self) -> None:
        self._reselect_task_button.enable()
        super(MMDetectionGUI, self).change_model()

    def set_deployed(self) -> None:
        self._reselect_task_button.disable()
        super(MMDetectionGUI, self).set_deployed()

    def get_ui(self) -> Widgets.Widget:
        return Widgets.Container(
            [
                Widgets.Container([
                    self._task_card,
                    self._models_card,
                ], gap=15),
                self._device_field,
                self._download_progress,
                self._success_label,
                self._serve_button,
                self._change_model_button,
            ],
            gap=3,
        )