# GUI event loops and toolkit integration for Python, most importantly
# to support plotting in a window without blocking.  Currently, we
# support wxWidgets, Qt4, and GTK+.

############################################################################

# global variable to specify default GUI toolkit to use
gui = :wx # one of :wx, :qt, or :gtk

pyexists(mod) = try
    pyimport(mod)
    true
  catch
    false
  end

pygui_works(gui::Symbol) =
    ((gui == :wx && pyexists("wx")) ||
     (gui == :gtk && pyexists("gtk")) ||
     (gui == :qt && (pyexists("PyQt4") || pyexists("PySide"))))
     
# get or set the default GUI; doesn't affect running GUI
function pygui()
    global gui
    if pygui_works(gui::Symbol)
        return gui::Symbol
    else
        for g in (:wx, :gtk, :qt)
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
        if g != :wx && g != :gtk && g != :qt
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
    timeout = Base.TimeoutAsyncWork(doevent)
    Base.start_timer(timeout,sec,sec)
    return timeout
end

# GTK:
function gtk_eventloop(sec::Real=50e-3)
    gtk = pyimport("gtk")
    events_pending = gtk["events_pending"]
    main_iteration = gtk["main_iteration"]
    function doevent(async, status::Int32) # handle all pending
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
    function doevent(async, status::Int32)
        app = pycall(instance, PyObject)
        if app.o != (pynothing::PyObject).o
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
    function doevent(async, status::Int32)
        app = pycall(GetApp, PyObject)
        if app.o != (pynothing::PyObject).o
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
const eventloops = (Symbol=>TimeoutAsyncWork)[]

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
        Base.stop_timer(delete!(eventloops, gui))
        true
    else
        false
    end
end

pygui_stop_all() = for gui in keys(eventloops); pygui_stop(gui); end

############################################################################
# Special support for matplotlib and pylab, to make them a bit easier to use,
# since you need to jump through some hoops to tell matplotlib which GUI to
# use and to employ interactive mode.

# map gui to corresponding matplotlib backend
const gui2matplotlib = [ :wx => "WXAgg", :gtk => "GTKAgg", :qt => "Qt4Agg" ]

function pymatplotlib(gui::Symbol=pygui())
    pygui_start(gui)
    m = pyimport("matplotlib")
    m[:use](gui2matplotlib[gui])
    m[:interactive](true)
    return m
end

# We monkey-patch pylab.show to ensure that it is non-blocking, as
# matplotlib does not reliably detect that our event-loop is running.
# (Note that some versions of show accept a "block" keyword or directly
# as a boolean argument, so we must accept the same arguments.)
function show_noop(b=false; block=false)
    nothing # no-op
end

function pylab()
    m = pyimport("pylab")
    m[:show] = show_noop
    m
end

macro pylab(optional_varname...)
    Name = pyimport_name(:pylab, optional_varname)
    quote
        if !isdefined($(Expr(:quote, Name)))
            pymatplotlib()
            const $(esc(Name)) = pywrap_module(pylab(), $(Expr(:quote, Name)))
        end
        nothing
    end
end

############################################################################
