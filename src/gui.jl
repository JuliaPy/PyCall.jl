# GUI event loops and toolkit integration for Python, most importantly
# to support plotting in a window without blocking.  Currently, we
# support wxWidgets, Qt5, Qt4, and GTK+.

############################################################################

# global variable to specify default GUI toolkit to use
gui = :default # one of :default, :wx, :qt, :tk, or :gtk

pyexists(mod) = try
    pyimport(mod)
    true
  catch
    false
  end

# Tkinter was renamed to tkinter in Python 3
function tkinter_name()
    return pyversion.major < 3 ? "Tkinter" : "tkinter"
end

pygui_works(gui::Symbol) = gui == :default ||
    ((gui == :wx && pyexists("wx")) ||
     (gui == :gtk && pyexists("gtk")) ||
     (gui == :gtk3 && pyexists("gi")) ||
     (gui == :tk && pyexists(tkinter_name())) ||
     ((gui == :qt5 || gui == :qt_pyqt5) && pyexists("PyQt5")) ||
     (gui == :qt_pyqt4 && pyexists("PyQt4")) ||
     (gui == :qt_pyside && pyexists("PySide")) ||
     (gui == :qt4 && (pyexists("PyQt4") || pyexists("PySide"))) ||
     (gui == :qt && (pyexists("PyQt5") || pyexists("PyQt4") || pyexists("PySide"))))

# get or set the default GUI; doesn't affect running GUI
"""
    pygui()

Return the current GUI toolkit as a symbol.
"""
function pygui()
    global gui
    if gui::Symbol != :default && pygui_works(gui::Symbol)
        return gui::Symbol
    else
        for g in (:tk, :qt, :wx, :gtk, :gtk3)
            if pygui_works(g)
                gui = g
                return gui::Symbol
            end
        end
        error("No supported Python GUI toolkit is installed.")
    end
end
function pygui(g::Symbol)
    global gui
    if g != gui::Symbol
        if !(g in (:wx,:gtk,:gtk3,:tk,:qt,:qt4,:qt5,:qt_pyqt5,:qt_pyqt4,:qt_pyside,:default))
            throw(ArgumentError("invalid gui $g"))
        elseif !pygui_works(g)
            error("Python GUI toolkit for $g is not installed.")
        end
        gui = g
    end
    return g::Symbol
end

############################################################################
# Event loops for various toolkits.

# call doevent(status) every sec seconds
function install_doevent(doevent, sec::Real)
    return Base.Timer(doevent, sec, sec)
end

# For PyPlot issue #181: recent pygobject releases emit a warning
# if we don't specify which version we want:
function gtk_requireversion(gtkmodule::AbstractString, vers::VersionNumber=v"3.0")
    if startswith(gtkmodule, "gi.")
        gi = pyimport("gi")
        if gi[:get_required_version]("Gtk") === nothing
            gi[:require_version]("Gtk", string(vers))
        end
    end
end

# GTK (either GTK2 or GTK3, depending on gtkmodule):
function gtk_eventloop(gtkmodule::AbstractString, sec::Real=50e-3)
    gtk_requireversion(gtkmodule)
    gtk = pyimport(gtkmodule)
    events_pending = gtk["events_pending"]
    main_iteration = gtk["main_iteration"]
    install_doevent(sec) do async
        # handle all pending
        while pycall(events_pending, Bool)
            pycall(main_iteration, PyObject)
        end
    end
end

# Qt4: (PyQt4 or PySide module)
function qt_eventloop(QtCore::PyObject, sec::Real=50e-3)
    instance = QtCore["QCoreApplication"]["instance"]
    AllEvents = QtCore["QEventLoop"]["AllEvents"]
    processEvents = QtCore["QCoreApplication"]["processEvents"]
    maxtime = PyObject(50)
    install_doevent(sec) do async
        app = pycall(instance, PyObject)
        if app.o != pynothing[]
            app["_in_event_loop"] = true
            pycall(processEvents, PyObject, AllEvents, maxtime)
        end
    end
end

qt_eventloop(QtModule::AbstractString, sec::Real=50e-3) =
    qt_eventloop(pyimport("$QtModule.QtCore"), sec)

function qt_eventloop(sec::Real=50e-3)
    for QtModule in ("PyQt5", "PyQt4", "PySide")
        try
            return qt_eventloop(QtModule, sec)
        end
    end
    error("no Qt module found")
end

# wx:  (based on IPython/lib/inputhookwx.py, which is 3-clause BSD-licensed)
function wx_eventloop(sec::Real=50e-3)
    wx = pyimport("wx")
    GetApp = wx["GetApp"]
    EventLoop = wx["EventLoop"]
    EventLoopActivator = wx["EventLoopActivator"]
    install_doevent(sec) do async
        app = pycall(GetApp, PyObject)
        if app.o != pynothing[]
            app["_in_event_loop"] = true
            evtloop = pycall(EventLoop, PyObject)
            ea = pycall(EventLoopActivator, PyObject, evtloop)
            Pending = evtloop["Pending"]
            Dispatch = evtloop["Dispatch"]
            while pycall(Pending, Bool)
                pycall(Dispatch, PyObject)
            end
            pydecref(ea) # deactivate event loop
            pycall(app["ProcessIdle"], PyObject)
        end
    end
end

# Tk: (Tkinter/tkinter module)
function Tk_eventloop(sec::Real=50e-3)
    Tk = pyimport(tkinter_name())
    install_doevent(sec) do async
        root = Tk["_default_root"]
        if root.o != pynothing[]
            pycall(root["update"], PyObject)
        end
    end
end
# cache running event loops (so that we don't start any more than once)
const eventloops = Dict{Symbol,Timer}()

"""
    pygui_start(gui::Symbol = pygui())

Start the event loop of a certain toolkit.

The argument `gui` defaults to the current default GUI, but it could be `:wx`, `:gtk`, `:gtk3`, `:tk`, or `:qt`.

"""
function pygui_start(gui::Symbol=pygui(), sec::Real=50e-3)
    pygui(gui)
    if !haskey(eventloops, gui)
        if gui == :wx
            eventloops[gui] = wx_eventloop(sec)
        elseif gui == :gtk
            eventloops[gui] = gtk_eventloop("gtk", sec)
        elseif gui == :gtk3
            eventloops[gui] = gtk_eventloop("gi.repository.Gtk", sec)
        elseif gui == :tk
            eventloops[gui] = Tk_eventloop(sec)
        elseif gui == :qt_pyqt5 || gui == :qt5
            eventloops[gui] = qt_eventloop("PyQt5", sec)
        elseif gui == :qt_pyqt4
            eventloops[gui] = qt_eventloop("PyQt4", sec)
        elseif gui == :qt_pyside
            eventloops[gui] = qt_eventloop("PySide", sec)
        elseif gui == :qt4
            try
                eventloops[gui] = qt_eventloop("PyQt4", sec)
            catch
                eventloops[gui] = qt_eventloop("PySide", sec)
            end
        elseif gui == :qt
            eventloops[gui] = qt_eventloop(sec)
        else
            throw(ArgumentError("unsupported GUI type $gui"))
        end
    end
    gui
end

"""
    pygui_stop(gui::Symbol = pygui())

Stop any running event loop for gui. The `gui` argument defaults to current default GUI.

"""
function pygui_stop(gui::Symbol=pygui())
    if haskey(eventloops, gui)
        Base.close(pop!(eventloops, gui))
        true
    else
        false
    end
end

pygui_stop_all() = for gui in keys(eventloops); pygui_stop(gui); end

############################################################################
