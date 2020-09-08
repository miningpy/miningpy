# Visualisation

## ParaView Glance Customisations

Specific functionalities were removed from ParaView Glance to make it more 
easily usable by end users. The following code snippets were deleted from the HTML file:

*,a("v-tooltip",{attrs:{bottom:"",disabled:e.smallScreen},scopedSlots:e._u([{key:"activator",fn:function(t){var i=t.on;return[a("a",e._g({attrs:{href:"#"},on:{click:function(t){return t.preventDefault(),e.toggleLanding(t)}}},i),[a("svg-icon",{staticStyle:{"margin-top":"6px"},attrs:{icon:"paraview-glance"+(e.smallScreen?"-small":""),height:"52px"}})],1)]}}],null,!0)},[e._v(" "),e.landingVisible?a("span",{key:"if-landingVisible"},[e._v("Go to app")]):a("span",{key:"if-landingVisible"},[e._v("Back to landing page")])])*

*,a("v-toolbar-items",[e.errors.length?a("v-btn",{key:"if-has-errors",attrs:{text:"",color:"error"},on:{click:function(t){e.errorDialog=!0}}},[a("v-icon",{attrs:{left:""}},[e._v("mdi-alert-circle")]),e._v(" "),a("span",[e._v(e._s(e.errors.length))]),e._v("\n             \n            "),a("span",{directives:[{name:"show",rawName:"v-show",value:!e.smallScreen,expression:"!smallScreen"}]},[e._v("error(s)")])],1):e._e()],1)*

*a("collapsible-toolbar-item",{attrs:{state:i},on:{click:e.showFileUpload}},[a("v-icon",{attrs:{left:""}},[e._v("mdi-folder")]),e._v(" "),a("span",{directives:[{name:"show",rawName:"v-show",value:"dense"!==i,expression:"state !== 'dense'"}]},[e._v("Open")])],1),*
