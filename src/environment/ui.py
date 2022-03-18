import keyboard


class UI:
    keys = {}

    @staticmethod
    def press(key):
        if type(key) is not str:
            key = key.name
        UI.keys[key] = True

    @staticmethod
    def release(key):
        if type(key) is not str:
            key = key.name
        UI.keys[key] = False

    @staticmethod
    def register_key(key, on_pressed=None, watch=True):
        UI.keys[key] = keyboard.is_pressed(key)
        if watch:
            keyboard.on_press_key(key, UI.press)
            keyboard.on_release_key(key, UI.release)
        if on_pressed is not None:
            keyboard.on_press_key(key, lambda _: on_pressed())

