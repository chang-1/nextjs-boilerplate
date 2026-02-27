declare module "react-plotly.js" {
  import * as React from "react";

  export interface PlotParams {
    data?: any[];
    layout?: any;
    frames?: any[];
    config?: any;
    style?: React.CSSProperties;
    className?: string;
    useResizeHandler?: boolean;
    onInitialized?: (figure: any, graphDiv: any) => void;
    onUpdate?: (figure: any, graphDiv: any) => void;
    onPurge?: (figure: any, graphDiv: any) => void;
    onError?: (err: any) => void;
    divId?: string;
    debug?: boolean;
    revision?: number;
  }

  const Plot: React.ComponentType<PlotParams>;
  export default Plot;
}
