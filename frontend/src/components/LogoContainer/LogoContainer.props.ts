import { AlignSelf } from "./LogoContainer.types";

export interface LogoContainerProps {
  /**
   * The content to be rendered inside the LogoContainer.
   */
  children: React.ReactNode;
  /**
   * The offset of the logo in the **number** of pixels.
   * If the value is `false`, it means it is centered.
   */
  logoOffset?: number;
  /**
   * The size of the logo in the number of px.
   */
  logoSize?: number;
  /**
   * @todo
   */
  onLogoClick?: () => void;
  /**
   * Defines how the logo is aligned along the cross axis.
   * Defaults to `center`.
   */
  alignSelf?: AlignSelf;
}
