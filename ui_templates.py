# Custom HTML

HTML_BANNER = """
    <div style="background-color:#080e4c;padding:10px;border-radius:5px">
    <h1 style="color:#3e47a5;text-align:center;">Voice Biometrics </h1>
    </div>
    """



HTML_BANNER_SKEWED = """
<style>
header {
  position: relative;
  height: 200px;
  background-image: linear-gradient(#3e47a5, #080e4c);
}

h2 {
  margin: 0;
  padding: 50px 0;
  font: 44px "Arial";
  text-align: center;
}

header h2 {
  color: gray;
}

.divider {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  width: 100%;
  height: 100px;
  /* drop the height to have a constant angle for all screen widths */
}
</style>
<header>
  <h2>BioVoice Tool</h2>
  <img src="http://erikdkennedy.com/r-r-r-random/divider-triangle.png" class="divider" />
</header>
"""

# https://code.tutsplus.com/tutorials/create-a-sticky-note-effect-in-5-easy-steps-with-css3-and-html5--net-13934
  
HTML_STICKER = """
<style>
*{
  margin:0;
  padding:0;
  border-radius:10px;
}
body{
  font-family:arial,sans-serif;
  font-size:100%;
  margin:1em;
  background:#3e47a5;
  color:#fff;
}
h2,p{
  font-size:100%;
  font-weight:normal;
}
ul,li{
  list-style:none;
}
ul{
  overflow:hidden;
  padding:3em;
}
ul li a{
  text-decoration:none;
  color:#000;
  background:#ffc;
  display:block;
  height:10em;
  width:10em;
  padding:1em;
  -moz-box-shadow:5px 5px 7px rgba(33,33,33,1);
  -webkit-box-shadow: 5px 5px 7px rgba(33,33,33,.7);
  box-shadow: 5px 5px 7px rgba(33,33,33,.7);
  -moz-transition:-moz-transform .15s linear;
  -o-transition:-o-transform .15s linear;
  -webkit-transition:-webkit-transform .15s linear;
}
ul li{
  margin:1em;
  float:left;
}
ul li h2{
  font-size:140%;
  font-weight:bold;
  padding-bottom:10px;
}
ul li p{
  font-family:"Reenie Beanie",arial,sans-serif;
  font-size:80%;
}
ul li a{
  -webkit-transform: rotate(-6deg);
  -o-transform: rotate(-6deg);
  -moz-transform:rotate(-6deg);
}
ul li:nth-child(even) a{
  -o-transform:rotate(4deg);
  -webkit-transform:rotate(4deg);
  -moz-transform:rotate(4deg);
  position:relative;
  top:5px;
  background:#cfc;
}
ul li:nth-child(3n) a{
  -o-transform:rotate(-3deg);
  -webkit-transform:rotate(-3deg);
  -moz-transform:rotate(-3deg);
  position:relative;
  top:-5px;
  background:#ccf;
}
ul li:nth-child(5n) a{
  -o-transform:rotate(5deg);
  -webkit-transform:rotate(5deg);
  -moz-transform:rotate(5deg);
  position:relative;
  top:-10px;
}
ul li a:hover,ul li a:focus{
  box-shadow:10px 10px 7px rgba(0,0,0,.7);
  -moz-box-shadow:10px 10px 7px rgba(0,0,0,.7);
  -webkit-box-shadow: 10px 10px 7px rgba(0,0,0,.7);
  -webkit-transform: scale(1.25);
  -moz-transform: scale(1.25);
  -o-transform: scale(1.25);
  position:relative;
  z-index:5;
}
ol{text-align:center;}
ol li{display:inline;padding-right:1em;}
ol li a{color:#fff;}

</style>
<ul>
    <li>
      <a>
        <h2>Prosody</h2>
        <p><b>Voice Fundamental Frequency Function<b/></p>
        <p>Mean</p>
        <p>Standard Deviation</p>
        <p>Median</p>
        <p>Min & Max</p>
        <p>25/75 Quantile</p>
      </a>
    </li>
    <li>
      <a>
        <h3>Gender Identification and Mood</h3>
        <br>
        <p><b>Gender ID</b></p>
        <p>(Female/Male)</p>
        <p><b>Speech Mode</b></p>
        <p>(Normal, Reading, Passionately)</p>
      </a>
    </li>
    <li>
      <a>
        <h3>Silent and Speech Rate</h3>
        <br>
        <p>Original Duration</p>
        <p>Speech Duration</p>
        <p>Number of pauses</p>
        <p>Speech/Pause Rate</p>
      </a>
    </li>
    <li>
      <a href="mailto:rosariomoscatolab@gmail.com"
        target="_blank" title="Send us an email">
        <h3>Voice Symilarity Level</h3>
        <br>
        <p>Very High</p>
        <p>High</p>
        <p>Medium</p>
        <p>Low</p>
        <br>
        <hr>
        <p><b>email</b>: rosariomoscatolab@gmail.com</p>
        <p></p>
        
        
      </a>
    </li>
    <li>
      <a href="https://rosariomoscato.github.io/"
        target="_blank" title="Link to Rosario Moscato Web">
        <h3>Voice Recognition and Transcription</h3>
        <br>
        <p><i>Multilanguage</i> Voice Recognition and Transcription to a text file</p>
        <br>
        <hr>
        <p><b>web</b>: rosariomoscato.github.io</p>
        <p></p>
    

      </a>
    </li>
    <li>
      <a href="https://www.youtube.com/channel/UCDn-FahQNJQOekLrOcR7-7Q"
        target="_blank" title="Link to YouTube Rosario Moscato LAB">
        <h2>About</h2>
        <p><b>Rosario Moscato</b></p>
        <p><i>AI Scientist</i></p>
        <br>
        <br>
        <br>
        <hr>

        <p><b>YouTube Channel</b>: RosarioMoscatoLAB</p>
      </a>
    </li>
  
  </ul>

"""

HTML_WRAPPER = """<div style="overflow: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""
