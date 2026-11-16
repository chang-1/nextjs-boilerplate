import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [{ source: "/jem-input-data/:path*", destination: "https://assets.jorgechang.xyz/seed1-input-gradients/data/:path*" }, { source: "/jem-data/:path*", destination: "https://assets.jorgechang.xyz/seed1-batch4/data/:path*" }];
  },
};

export default nextConfig;
