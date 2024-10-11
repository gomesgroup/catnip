import {ResponsiveScatterPlot} from '@nivo/scatterplot'
import {animated} from '@react-spring/web'
import MoleculeStructure from "./MoleculeStructure/MoleculeStructure";
import {useCallback} from 'react'
import {useMemo} from 'react'


function getGradientColor(x, y, width, height) {
    let theta = Math.atan2(-y, x);
    if (theta < 0) {
        theta += 2 * Math.PI;
    }

    let hue = theta * (180 / Math.PI);

    let finalColor = hsvToRgb(hue, 1, 1);

    return `rgb(${Math.round(finalColor.r)}, ${Math.round(finalColor.g)}, ${Math.round(finalColor.b)})`;
}

function hsvToRgb(h, s, v) {
    let r, g, b;
    let i = Math.floor(h / 60);
    let f = h / 60 - i;
    let p = v * (1 - s);
    let q = v * (1 - s * f);
    let t = v * (1 - s * (1 - f));

    switch (i % 6) {
        case 0:
            r = v;
            g = t;
            b = p;
            break;
        case 1:
            r = q;
            g = v;
            b = p;
            break;
        case 2:
            r = p;
            g = v;
            b = t;
            break;
        case 3:
            r = p;
            g = q;
            b = v;
            break;
        case 4:
            r = t;
            g = p;
            b = v;
            break;
        case 5:
            r = v;
            g = p;
            b = q;
            break;
    }

    return {
        r: Math.round(r * 255),
        g: Math.round(g * 255),
        b: Math.round(b * 255)
    };
}


export const CustomColoredNode = ({largeDotCoordinates = [], ...props}) => {

    const handleMouseEnter = useCallback(event =>
            props.onMouseEnter?.(props.node, event),
        [props.node, props.onMouseEnter]
    )

    const handleMouseMove = useCallback(event =>
            props.onMouseMove?.(props.node, event),
        [props.node, props.onMouseMove]
    )

    const handleMouseLeave = useCallback(event =>
            props.onMouseLeave?.(props.node, event),
        [props.node, props.onMouseLeave]
    )

    const isLargeDot =
        largeDotCoordinates.length >= 2 &&
        props.node.data.x === largeDotCoordinates[0] &&
        props.node.data.y === largeDotCoordinates[1];

    return (
        <animated.circle
            cx={props.style.x}
            cy={props.style.y}
            r={isLargeDot ? props.style.size.to(size => size) : props.style.size.to(size => size / 2)}
            fill={isLargeDot ? 'white' : getGradientColor(props.node.xValue, props.node.yValue, 6, 6)}
            style={{mixBlendMode: props.blendMode}}
            onMouseEnter={handleMouseEnter}
            onMouseLeave={handleMouseLeave}
            onMouseMove={handleMouseMove}
        />
    );
};

function ScatterPlot({data, largeDotCoordinates, largeDotLegend}) {
    const dataWithLargeDot = useMemo(() => {
        if (largeDotCoordinates && largeDotCoordinates.length >= 2) {
            return [
                ...data,
                {
                    id: largeDotLegend,
                    data: [{x: largeDotCoordinates[0], y: largeDotCoordinates[1], smiles: largeDotLegend}]
                },
            ];
        } else {
            return data;
        }
    }, [data, largeDotCoordinates, largeDotLegend]);

    return (
        <ResponsiveScatterPlot
            data={dataWithLargeDot}
            margin={{right: 10, bottom: 70, left: 60, top: 10}}
            xScale={{type: 'linear', min: -7.2, max: 11}}
            xFormat=">-.2f"
            yScale={{type: 'linear', min: -7.2, max: 11}}
            yFormat=">-.2f"
            axisTop={null}
            axisRight={null}
            nodeComponent={props => <CustomColoredNode largeDotCoordinates={largeDotCoordinates} {...props} />}
            theme={{
                axis: {
                    "domain": {
                        "line": {
                            "stroke": "#777777",
                            "strokeWidth": 1
                        }
                    },

                    ticks: {
                        text: {
                            fill: "#ffffff",
                            fontSize: 8,
                        }
                    },

                    legend: {
                        text: {
                            fill: "#ffffff",
                            fontSize: 12,
                        }
                    }

                },
                grid: {
                    line: {
                        "strokeWidth": 0
                    }
                },

            }}
            axisBottom={{
                orient: 'bottom',
                tickSize: 5,
                tickPadding: 5,
                tickRotation: 0,
                legend: 'PC1',
                legendPosition: 'middle',
                legendOffset: 40
            }}
            axisLeft={{
                orient: 'left',
                tickSize: 5,
                tickPadding: 5,
                tickRotation: 0,
                legend: 'PC2',
                legendPosition: 'middle',
                legendOffset: -40
            }}
            useMesh={false}
            tooltip={(node) => {
                console.log(node);

                return (
                    <div
                        className="animate-appear" // this is a hack to stop the tooltip from flickering
                        style={{
                            background: "white",
                            padding: "3px 3px",
                            border: "1px solid #000",
                        }}
                    >
                        <MoleculeStructure id={node.node.index} structure={node.node.data.smiles} width={100}
                                           height={100}/>
                    </div>
                );
            }}
        />);
}

export default ScatterPlot;
