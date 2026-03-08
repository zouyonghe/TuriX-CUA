import platform


SYSTEM_NAME = platform.system()

if SYSTEM_NAME == "Windows":
    from src.windows.actions import WindowsActions as PlatformActions
    from src.windows.openapp import list_applications, open_application_by_name
elif SYSTEM_NAME == "Linux":
    from src.linux.actions import LinuxActions as PlatformActions
    from src.linux.openapp import list_applications, open_application_by_name
else:
    raise NotImplementedError(
        f"Unsupported OS '{SYSTEM_NAME}'. This branch currently supports Windows and Linux."
    )

