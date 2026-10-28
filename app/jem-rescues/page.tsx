import type { Metadata } from "next";
import Explorer from "./Explorer";

export const metadata: Metadata = {
  title: "JEM | ImageNet Hybrid Rescue Explorer",
  description:
    "Explore ImageNet validation examples rescued or lost by hybrid JEM objectives compared with alpha 0.",
};

export default function JemRescuesPage() {
  return <Explorer />;
}
