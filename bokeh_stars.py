import numpy as np

from jinja2 import Template

from bokeh.io import vform
from bokeh.models import CustomJS, ColumnDataSource, Slider, HoverTool
from bokeh.plotting import figure # , output_file, show
from bokeh.models.widgets.layouts import HBox
from bokeh.embed import components
from bokeh.resources import Resources

n_stars = 50
yags = np.random.uniform(-2400, 2400, size=n_stars)
zags = np.random.uniform(-2400, 2400, size=n_stars)
colors = ["navy"] * n_stars
alphas = 0.5 * np.ones(n_stars)
mags = np.random.uniform(6, 12, size=n_stars)
sizes = 14 * np.ones(n_stars) - mags
yag_zag_strs = ['{:.1f} {:.1f}'.format(yag, zag) for yag, zag in zip(yags, zags)]

stars = ColumnDataSource(data=dict(yag=yags, zag=zags, color=colors,
                                   alpha=alphas, mag=mags, size=sizes,
                                   yag_zag_str=yag_zag_strs))
# 

hover = HoverTool(
        tooltips=[
            ("yag,zag", "@yag_zag_str"),
            ("mag", "@mag"),
        ]
    )

plot = figure(plot_width=400, plot_height=400, x_range=[2500, -2500], y_range=[-2500, 2500],
              tools=[hover, 'box_zoom', 'pan', 'reset'])
plot.yaxis.major_label_orientation = "vertical"

plot.circle('yag', 'zag', color='color', size='size', alpha='alpha', source=stars)


slider = Slider(start=6, end=12, value=12, step=.1, title="Mag limit")
slider.callback = CustomJS(args=dict(source=stars), code="""
        var data = source.get('data');
        var f = cb_obj.get('value');
        var alpha = 0.0;

        // window.alert(Object.keys(data));
        for (i = 0; i < data['size'].length; i++) {
            if (data['mag'][i] > f) {
                data['alpha'][i] = 0.0;
            } else {
                data['alpha'][i] = 0.5;
            }
        }

        source.trigger('change');
    """)


layout = vform(HBox(slider), plot)

resources = Resources('inline')  # Probably 'inline' for production
script, div = components(plot, resources)

with open('bokeh_stars_template.html', 'rb') as fh:
    template = Template(fh.read())

out = template.render(bokeh_js='\n'.join(resources.js_raw),
                      bokeh_css='\n'.join(resources.css_raw),
                      script=script, div=div)

with open('bokeh_stars.html', 'wb') as fh:
    fh.write(out)
