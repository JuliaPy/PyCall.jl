# GUI event loops and toolkit integration for Python, most importantly
# to support plotting in a window without blocking.  Currently, we
# support wxWidgets, Qt4, and GTK+.

############################################################################

# global variable to specify default GUI toolkit to use
gui = :default # one of :default, :wx, :qt, or :gtk

pyexists(mod) = try
    pyimport(mod)
    true
  catch
    false
  end

pygui_works(gui::Symbol) = gui == :default ||
    ((gui == :wx && pyexists("wx")) ||
     (gui == :gtk && pyexists("gtk")) ||
     (gui == :qt && (pyexists("PyQt4") || pyexists("PySide"))))
     
# get or set the default GUI; doesn't affect running GUI
function pygui()
    global gui
    if gui::Symbol != :default && pygui_works(gui::Symbol)
        return gui::Symbol
    else
        for g in (:qt, :wx, :gtk)
            if pygui_works(g)
                gui::Symbol = g
                return gui::Symbol
            end
        end
        error("No supported Python GUI toolkit is installed.")
    end
end
function pygui(g::Symbol)
    global gui
    if g != gui::Symbol
        if g != :wx && g != :gtk && g != :qt && g != :default
            throw(ArgumentError("invalid gui $g"))
        elseif !pygui_works(g)
            error("Python GUI toolkit for $g is not installed.")
        end
        gui::Symbol = g
    end
    return g
end

############################################################################
# Event loops for various toolkits.

# call doevent(status) every sec seconds
function install_doevent(doevent::Function, sec::Real)
    timeout = Base.Timer(doevent)
    Base.start_timer(timeout,sec,sec)
    return timeout
end

# work around API change in Julia 0.3 (issue #76)
if VERSION >= v"0.3-"
    macro doevent(body)
        Expr(:function, :($(esc(:doevent))(async)), body)
    end
else
    macro doevent(body)
        Expr(:function, :($(esc(:doevent))(async, status::Int32)), body)
    end
end

# GTK:
function gtk_eventloop(sec::Real=50e-3)
    gtk = pyimport("gtk")
    events_pending = gtk["events_pending"]
    main_iteration = gtk["main_iteration"]
    @doevent begin
        # handle all pending
        while pycall(events_pending, Bool)
            pycall(main_iteration, PyObject)
        end
    end
    install_doevent(doevent, sec)
end

# Qt4: (PyQt4 or PySide module)
function qt_eventloop(QtModule="PyQt4", sec::Real=50e-3)
    QtCore = pyimport("$QtModule.QtCore")
    instance = QtCore["QCoreApplication"]["instance"]
    AllEvents = QtCore["QEventLoop"]["AllEvents"]
    processEvents = QtCore["QCoreApplication"]["processEvents"]
    maxtime = PyObject(50)
    @doevent begin
        app = pycall(instance, PyObject)
        if app.o != pynothing::PyPtr
            app["_in_event_loop"] = true
            pycall(processEvents, PyObject, AllEvents, maxtime)
        end
    end
    install_doevent(doevent, sec)
end

# wx:  (based on IPython/lib/inputhookwx.py, which is 3-clause BSD-licensed) 
function wx_eventloop(sec::Real=50e-3)
    wx = pyimport("wx")
    GetApp = wx["GetApp"]
    EventLoop = wx["EventLoop"]
    EventLoopActivator = wx["EventLoopActivator"]
    @doevent begin
        app = pycall(GetApp, PyObject)
        if app.o != pynothing::PyPtr
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
    install_doevent(doevent, sec)
end

# cache running event loops (so that we don't start any more than once)
const eventloops = Dict{Symbol,Timer}()

function pygui_start(gui::Symbol=pygui(), sec::Real=50e-3)
    pygui(gui)
    if !haskey(eventloops, gui)
        if gui == :wx
            eventloops[gui] = wx_eventloop(sec)
        elseif gui == :gtk
            eventloops[gui] = gtk_eventloop(sec)
        elseif gui == :qt
            try
                eventloops[gui] = qt_eventloop("PyQt4", sec)
            catch
                eventloops[gui] = qt_eventloop("PySide", sec)
            end
        else
            throw(ArgumentError("unsupported GUI type $gui"))
        end
    end
    gui
end

function pygui_stop(gui::Symbol=pygui())
    if haskey(eventloops, gui)
        Base.stop_timer(pop!(eventloops, gui))
        true
    else
        false
    end
end

pygui_stop_all() = for gui in keys(eventloops); pygui_stop(gui); end

############################################################################
