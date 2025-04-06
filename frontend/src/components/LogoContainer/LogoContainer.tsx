import Logo from "@components/Logo/Logo";

import { LogoContainerProps } from "./LogoContainer.props";

import classes from "./LogoContainer.module.css";

const LogoContainer: React.FC<LogoContainerProps> = ({
  children,
  logoOffset,
}) => {
  const logoStylesOfOffset = logoOffset
    ? { alignSelf: "start", marginLeft: `${logoOffset}px` }
    : undefined;

  return (
    <div className={classes.LogoContainer}>
      <Logo style={logoStylesOfOffset} />
      {children}
    </div>
  );
};

export default LogoContainer;
