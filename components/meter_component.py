import streamlit.components.v1 as components

def meter_component(green, yellow, red, metrics=None, width=200, height=200):
    """Render the circular meter visualization with metrics display"""
    html = f"""
    <div style="display: flex; flex-direction: column; align-items: flex-start; gap: 10px;">
        {f'<div class="metrics" style="text-align: left; font-size: 14px; color: #666; width: {width}px;">{metrics}</div>' if metrics else ''}
        <div class="meter-container">
        <svg class="meter" viewBox="0 0 86 86" width="{width}" height="{height}">
            <defs>
                <filter id="shadow" x="-50%" y="-50%" height="200%" width="200%">
                    <feOffset in="SourceAlpha" dx="0" dy="4" result="offset"/>
                    <feColorMatrix in="offset" type="matrix" 
                        values="0 0 0 0.4 0 0 0 0 0.4 0 0 0 0 0.4 0 0 0 0 0.4 0" 
                        result="matrix"/>
                    <feGaussianBlur in="matrix" stdDeviation="3" result="blur"/>
                    <feBlend in="SourceGraphic" in2="blur" mode="normal"/>
                </filter>
            </defs>

            <g>
                <circle cx="50%" cy="50%" r="26.25" 
                    fill="white" 
                    stroke-width="8" 
                    stroke="none" 
                    class="shadow-filter"/>
            </g>

            <g transform="rotate(180, 43, 43)">
                <circle cx="50%" cy="50%" r="35" 
                    stroke-dasharray="109.96, 109.96" 
                    fill="none" 
                    stroke-width="3.5" 
                    stroke="#E5E5E5" 
                    stroke-linecap="round"/>
                <circle id="redArc" cx="50%" cy="50%" r="35"
                    stroke-dasharray="109.96, 109.96"
                    fill="none"
                    stroke-width="3.5"
                    stroke="#FA5B02"
                    stroke-linecap="round"/>
                <circle id="yellowArc" cx="50%" cy="50%" r="35"
                    fill="none"
                    stroke-width="3.5"
                    stroke="#FFB800"
                    stroke-linecap="round"/>
                <circle id="greenArc" cx="50%" cy="50%" r="35"
                    fill="none"
                    stroke-width="3.5"
                    stroke="#3AE287"
                    stroke-linecap="round"/>
            </g>

            <text x="50%" y="52%" 
                text-anchor="middle" 
                dominant-baseline="middle" 
                font-size="15.75px" 
                fill="#3AE287"
                font-family="Orkney, Arial" 
                font-weight="600" 
                id="percentText">{green}%</text>
        </svg>
    </div>

    <script>
        function updateMeter(green, yellow, red) {{
            const circumference = 2 * Math.PI * 35;
            const halfCirc = circumference / 2;

            const greenArc = document.getElementById('greenArc');
            const yellowArc = document.getElementById('yellowArc');
            const redArc = document.getElementById('redArc');
            const percentText = document.getElementById('percentText');

            // Set arc lengths
            greenArc.setAttribute('stroke-dasharray', 
                `${{(green / 100) * halfCirc}}, ${{circumference}}`);
            yellowArc.setAttribute('stroke-dasharray',
                `${{((green + yellow) / 100) * halfCirc}}, ${{circumference}}`);
            redArc.setAttribute('stroke-dasharray',
                `${{halfCirc}}, ${{halfCirc}}`);

            // Update text
            percentText.textContent = `${{Math.round(green)}}%`;
        }}

        // Initial update
        updateMeter({green}, {yellow}, {red});
    </script>
        </div>
    </div>
    """
    
    components.html(html, height=height)
